import os, re, sys, time, asyncio, hashlib, json, aiohttp, backoff
import importlib
from io import BytesIO
from PIL import Image
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Any
from lxml import etree
from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# --- ВШИТЫЙ ПАТТЕРН ---


class Pattern:
    def match_score(self, url: str, html_content: str, raw_product: dict) -> float:
        """Оценивает (от 0 до 1), насколько сайт похож на автозапчасти"""
        text_lower = html_content.lower()
        score = 0.0
        if any(w in text_lower for w in ["б/у", "пробег", "контрактн", "двигатель"]): score += 0.8
        return min(score, 1.0)
        
    def get_custom_labels(self, html_content: str, brand: str, cat_name: str) -> list[str]:
        custom_labels = []
        if brand and brand != "Unknown": custom_labels.append(brand)
        if cat_name and cat_name != "Каталог": custom_labels.append(cat_name)
        page_text_lower = html_content.lower()
        if any(w in page_text_lower for w in ["б/у", "бывш", "пробег"]): custom_labels.append("Б/У")
        if "контрактн" in page_text_lower: custom_labels.append("Контрактный")
        return custom_labels[:5]

    def clean_description(self, text: str) -> str:
        fluff_patterns = [
            r'(?i)комментарий от продавца[:\-\s]*', r'(?i)внимание[:\-\s]*', r'(?i)уважаемые покупатели[:\-\s]*',
            r'(?i)мы предоставляем полный пакет документов.*?учёт[:\-\s]*', r'(?i)копия грузовой.*?деклараци[ии].*?(?=\.|\n|$)',
            r'(?i)договор купли-продажи.*?(?=\.|\n|$)', r'(?i)есть аукционный лист.*?(?=\.|\n|$)',
            r'(?i)предоставим подробное фото.*?видео.*?(?=\.|\n|$)', r'(?i)возможна проверка эндоскопом.*?(?=\.|\n|$)',
            r'(?i)цена указана за.*?фото.*?(?=\.|\n|$)', r'(?i)описание товара[:\-\s]*',
            r'(?i)возможна продажа без навесного.*?([.\n]|$)', r'(?i)возможна продажа.*?([.\n]|$)',
            r'(?i)(Номер по производителю|Производитель|Марка|Модель|Год|Кузов|Артикул)[\s:]*$' 
        ]
        for pattern in fluff_patterns: 
            text = re.sub(pattern, '', text)
        return text

    def generate_keywords(self, type_prefix: str, specs: dict) -> str:
        stop_keys = ['марка', 'бренд', 'производитель', 'модель']
        safe_specs = [str(v) for k, v in specs.items() if str(k).lower() not in stop_keys]
        specs_str = " ".join(safe_specs)
        # Формируем структуру: [Тип] товар [Характеристики без бренда/модели]
        return f"{type_prefix} товар {specs_str}".strip()
# ----------------------

class RawExtractedProduct(BaseModel):
    h1_title: str = Field(default="Без названия")
    brand: str = Field(default="Unknown")
    price_raw: Any = Field(default=0, alias="price")
    oldprice_raw: Any = Field(default=0, alias="oldprice")
    currency: str = Field(default="RUB")
    images: list[str] = Field(default_factory=list)
    specs: dict[str, Any] = Field(default_factory=dict)
    available: bool = Field(default=True)
    category_name: str = Field(default="Каталог")
    category_usp: str = Field(default="")
    description_usp: str = Field(default="")
    sales_notes: str = Field(default="")
    custom_labels: list[str] = Field(default_factory=list)
    variations: list[dict] = Field(default_factory=list)
    ai_templates: dict = Field(default_factory=dict)
    selectors: dict = Field(default_factory=dict)
    semantic_pattern: str = Field(default="")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class TransformedProduct(BaseModel):
    offer_id: str
    url: str
    name: str
    type_prefix: str
    vendor: str
    model_name: str
    price: str
    oldprice: str
    currency: str
    images: list[str]
    description: str
    sales_notes: str
    specs: dict[str, str]
    custom_labels: list[str]
    available: str
    category_id: str


class CategoryCollection(BaseModel):
    category_id: str
    name: str
    url: str
    picture: str
    description: str

# ==========================================
# СИСТЕМА ДИНАМИЧЕСКИХ ПАТТЕРНОВ (ENSEMBLE)
# ==========================================


class PatternManager:
    def __init__(self, patterns_dir="patterns"):
        self.patterns = []
        if not os.path.exists(patterns_dir): return
        
        for filename in os.listdir(patterns_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"patterns.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and hasattr(attr, 'match_score'):
                            self.patterns.append(attr())
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки паттерна {filename}: {e}")

    def apply_best_patterns(self, url: str, html_content: str, raw_product: RawExtractedProduct) -> RawExtractedProduct:
        if not self.patterns: return raw_product

        scored_patterns = []
        for p in self.patterns:
            try:
                score = p.match_score(url, html_content, raw_product)
                if score > 0.1: scored_patterns.append((score, p))
            except: pass

        if not scored_patterns: return raw_product

        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        top_patterns = [p for score, p in scored_patterns[:2]]

        desc = raw_product.description_usp
        for p in top_patterns:
            if hasattr(p, 'clean_description'):
                desc = p.clean_description(desc)
        raw_product.description_usp = desc

        labels_set = set(raw_product.custom_labels)
        for p in top_patterns:
            if hasattr(p, 'get_custom_labels'):
                new_labels = p.get_custom_labels(html_content, raw_product.brand, raw_product.category_name)
                labels_set.update(new_labels)
        raw_product.custom_labels = list(labels_set)[:5]

        return raw_product

# ==========================================
# УТИЛИТА: ВАЛИДАЦИЯ ИЗОБРАЖЕНИЙ
# ==========================================

class CacheManager:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}

    def save(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)

    def generate_fingerprint(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, 'lxml')
        h1_el = soup.find(['h1', 'h2'], class_=re.compile(r'title|name|head', re.I)) or soup.find('h1')
        h1_title = h1_el.get_text(strip=True) if h1_el else "unknown_item"

        price_val = "0"
        price_regex = r'(?<!\d)(\d[\d\s.,\xa0]{0,15})\s*(₽|руб|rub|р\.|р|\$|usd|€|eur|₸|kzt|byn)'
        for node in soup.find_all(class_=re.compile(r'price|cost|amount', re.I)):
            matches = re.findall(price_regex, node.get_text(separator=' ').lower(), re.IGNORECASE)
            for val_str, _ in matches:
                clean_val = re.sub(r'[^\d]', '', val_str)
                if clean_val and int(clean_val) > 0:
                    price_val = clean_val
                    break
            if price_val != "0": break

        page_text_lower = html_content.lower()
        available = "false" if any(w in page_text_lower for w in ["нет в наличии", "out of stock", "под заказ"]) else "true"
        return hashlib.md5(f"{h1_title}_{price_val}_{available}".encode('utf-8')).hexdigest()

    def check_cache(self, url: str, html_content: str) -> tuple[bool, Optional[dict]]:
        if not html_content:
            return (True, self.cache[url].get("raw_data")) if url in self.cache else (False, None)
        current_fingerprint = self.generate_fingerprint(html_content)
        if url in self.cache and self.cache[url].get("fingerprint") == current_fingerprint:
            return True, self.cache[url].get("raw_data")
        return False, None

    def update_cache(self, url: str, html_content: str, raw_data: dict):
        self.cache[url] = {"fingerprint": self.generate_fingerprint(html_content), "raw_data": raw_data, "last_seen": datetime.now().isoformat()}
        self.save()

    def patch_cache(self, url: str, new_price: str, new_oldprice: str, is_available: bool):
        if url in self.cache:
            self.cache[url]["raw_data"].update({"price": new_price, "oldprice": new_oldprice, "available": is_available})
            self.cache[url]["last_seen"] = datetime.now().isoformat()
            self.save()
            return True
        return False

    def get_all_cached_urls(self): return set(self.cache.keys())
    def get_raw_data(self, url: str): return self.cache.get(url, {}).get("raw_data")

    def get_few_shot_examples(self, domain: str, limit: int = 2) -> list:
        examples = []
        for url, data in self.cache.items():
            if domain in url and "raw_data" in data and data["raw_data"].get("h1_title") not in [None, "Без названия"]:
                rd = data["raw_data"]
                examples.append({"name": rd["h1_title"], "description": rd.get("description_usp"), "sales_notes": rd.get("sales_notes")})
                if len(examples) >= limit: break
        return examples

# ==========================================
# 3. ФАЗА РАЗВЕДКИ (DISCOVERY)
# ==========================================

class DiscoveryAgent:
    @staticmethod
    def analyze_and_group_links(base_url: str, html_content: str) -> dict:
        groups = {}
        base_parsed = urlparse(base_url)
        soup = BeautifulSoup(html_content, 'lxml')
        stop_words = ['login', 'cart', 'korzina', 'tel:', 'mailto:', '.jpg', '.png', 'policy', 'consent', 'contacts', 'kontakt', 'pro-o-nas', 'about', 'oplata', 'dostavka', 'rezerv', 'vozvrat', 'otzyvy', 'faq', 'help', 'garantiya', '+7', '8800', 'javascript:', 'whatsapp', 'viber', 'tg://', 'auth', 'register']

        for a in soup.find_all('a', href=True):
            href = a.get('href')
            parsed_href = urlparse(href)
            if (parsed_href.netloc and parsed_href.netloc != base_parsed.netloc) or any(w in parsed_href.path.lower() for w in stop_words): continue

            clean_href = href.split('#')[0].split('?')[0]
            if clean_href in ['/', '', base_parsed.netloc]: continue

            full_url = urljoin(base_url, clean_href)
            path_parts = [p for p in parsed_href.path.lower().split('/') if p]
            signature = f"/{'/'.join(path_parts[:-1])}/*" if len(path_parts) > 1 else f"/{path_parts[0]}/*" if path_parts else "/"

            title = ""
            parent_card = a.find_parent(['div', 'li', 'article'], class_=lambda c: c and any(x in c.lower() for x in ['product', 'item', 'good', 'card']))
            if parent_card:
                texts = [t for t in parent_card.stripped_strings if t]
                if texts: title = " | ".join(texts[:3])

            if not title:
                title = a.get_text(strip=True) or a.get('title', '')
                if not title and a.find('img'): title = a.find('img').get('alt', '') or a.find('img').get('title', '')

            title = ' '.join(title.split()) if title else "Без названия"
            if signature not in groups: groups[signature] = {}
            if full_url not in groups[signature] or len(title) > len(groups[signature].get(full_url, "")): groups[signature][full_url] = title

        result = {sig: [{"url": k, "title": v} for k, v in links.items()] for sig, links in groups.items() if links}
        return dict(sorted(result.items(), key=lambda i: (1 if any(x in i[0] for x in ['product', 'item', 'detail', 'catalog/']) else 0, len(i[1])), reverse=True))

# ==========================================
# 4. ФАЗА ИЗВЛЕЧЕНИЯ (КЛАССИКА И ИИ)
# ==========================================

class ClassicScraper:
    @staticmethod
    def extract_product_data(url: str, html_content: str, markdown_content: str = "", domain_rules: dict = None) -> RawExtractedProduct:
        soup = BeautifulSoup(html_content, 'lxml')
        h1_title = ""
        price_raw = "0"
        desc_usp = ""
        images = []

        selectors = domain_rules.get("selectors", {}) if domain_rules else {}
        if selectors:
            try:
                if selectors.get("h1_title"):
                    el = soup.select_one(selectors["h1_title"])
                    if el: h1_title = el.get_text(strip=True)
                if selectors.get("price"):
                    el = soup.select_one(selectors["price"])
                    if el:
                        clean_val = re.sub(r'[^\d]', '', el.get_text())
                        if clean_val: price_raw = clean_val
                if selectors.get("description"):
                    el = soup.select_one(selectors["description"])
                    if el: desc_usp = el.get_text(separator=' ', strip=True)
                if selectors.get("images"):
                    img_els = soup.select(selectors["images"])
                    for img in img_els:
                        src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                        if src and src.startswith('http'): images.append(src)
            except Exception: pass

        if not h1_title:
            h1_el = soup.find(['h1', 'h2'], class_=re.compile(r'title|name|head', re.I)) or soup.find('h1')
            h1_title = h1_el.get_text(strip=True) if h1_el else "Без названия"

        currency = "RUB"
        currency_map = {'₽': 'RUB', 'руб': 'RUB', 'rub': 'RUB', 'р.': 'RUB', 'р': 'RUB', '$': 'USD', 'usd': 'USD', '€': 'EUR', 'eur': 'EUR', '₸': 'KZT', 'kzt': 'KZT', 'byn': 'BYN'}
        price_regex = r'(?<!\d)(\d[\d\s.,\xa0]{0,15})\s*(₽|руб|rub|р\.|р|\$|usd|€|eur|₸|kzt|byn)'

        if price_raw == "0":
            def find_price() -> tuple[str, str]:
                if markdown_content:
                    clean_md = markdown_content.replace('&nbsp;', '').replace('&#160;', '')
                    matches_md = re.findall(price_regex, clean_md.lower(), re.IGNORECASE)
                    for val_str, curr_str in matches_md:
                        clean_val = re.sub(r'[^\d]', '', val_str)
                        if clean_val and int(clean_val) > 0:
                            return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')

                for node in soup.find_all(class_=re.compile(r'price|cost|amount', re.I)):
                    text = node.get_text(separator=' ').lower()
                    matches = re.findall(price_regex, text, re.IGNORECASE)
                    for val_str, curr_str in matches:
                        clean_val = re.sub(r'[^\d]', '', val_str)
                        if clean_val and int(clean_val) > 0:
                            return clean_val, currency_map.get(curr_str.strip().lower(), 'RUB')
                return "0", "RUB"
            price_raw, currency = find_price()

        oldprice_raw = "0"
        for oldprice_node in soup.find_all(class_=re.compile(r'old-price|old_price|price-old|crossed', re.I)):
            clean_str = re.sub(r'[^\d]', '', oldprice_node.get_text())
            if clean_str and int(clean_str) > 0:
                oldprice_raw = clean_str
                break

        if not images:
            for img_container in soup.find_all(['div', 'a', 'img', 'picture'], class_=re.compile(r'img|image|slider|gallery|photo', re.I)):
                img = img_container if img_container.name == 'img' else img_container.find('img')
                if img:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy')
                    if src and src.startswith('http'): images.append(src)

        specs = {}
        brand = "Unknown"

        for row in soup.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            if len(cols) == 2:
                specs[cols[0].get_text(strip=True)] = cols[1].get_text(strip=True)

        for item in soup.find_all(['li', 'div'], class_=re.compile(r'param|property|feature|attribute', re.I)):
            name_el = item.find(['div', 'span'], class_=re.compile(r'name|title|label', re.I))
            val_el = item.find(['div', 'span'], class_=re.compile(r'value|val', re.I))
            if name_el and val_el:
                specs[name_el.get_text(strip=True)] = val_el.get_text(strip=True)
            else:
                text = item.get_text(strip=True)
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts[0]) < 30:
                        specs[parts[0].strip()] = parts[1].strip()

        for k, v in specs.items():
            if k.lower() in ['марка', 'производитель', 'бренд']:
                brand = v

        if not desc_usp:
            desc_container = soup.find(['div', 'section'], class_=re.compile(r'comment|description|detail-text|about', re.I))
            if desc_container:
                desc_usp = desc_container.get_text(separator=' ', strip=True)

        sales_notes = ""
        delivery_nodes = soup.find_all(string=re.compile(r'(предоплат|оплат|картой|доставк|тк |отправк|гарант|возврат|срок)', re.I))
        for node in delivery_nodes:
            parent = getattr(node, 'parent', None)
            if parent and parent.name not in ['script', 'style']:
                text = re.sub(r'\s+', ' ', parent.get_text(separator=' ', strip=True))
                if re.search(r'\d+', text) and 5 < len(text) <= 50:
                    sales_notes = text
                    break

        cat_name = "Каталог"
        breadcrumbs = soup.find_all(['span', 'li', 'a', 'div'], class_=re.compile(r'breadcrumb|bx-breadcrumb|nav', re.I))
        if len(breadcrumbs) > 1:
            cat_name = breadcrumbs[-1].get_text(strip=True)

        custom_labels = []
        if brand != "Unknown": custom_labels.append(brand)
        if cat_name != "Каталог": custom_labels.append(cat_name)

        page_text_lower = html_content.lower()
        if "б/у" in page_text_lower or "бывш" in page_text_lower or "пробег" in page_text_lower:
            custom_labels.append("Б/У")
        if "контрактн" in page_text_lower:
            custom_labels.append("Контрактный")

        return RawExtractedProduct(
            h1_title=h1_title, brand=brand, price_raw=price_raw, oldprice_raw=oldprice_raw,
            currency=currency, images=list(set(images)), specs=specs, category_name=cat_name,
            category_usp=cat_name, description_usp=desc_usp, sales_notes=sales_notes, custom_labels=custom_labels[:5], variations=[], ai_templates={}
        )


class DataTransformer:
    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def clean_punctuation(text: str) -> str:
        text = str(text)
        # Убираем дублирование знаков препинания на стыках, образующихся от склеивания шаблонов
        text = re.sub(r'[,;]\s*\.', '.', text)
        text = re.sub(r'\.\s*,', '.', text)
        text = re.sub(r'\.{2,}', '.', text)
        # Исправляем лишние пробелы перед знаками
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        # Удаляем "висячую" пунктуацию В НАЧАЛЕ строки
        text = re.sub(r'^[\s.,!?;:\-]+', '', text)
        # Удаляем лишнюю "висячую" пунктуацию В КОНЦЕ строки (оставляя нормальную, если нужно, но в нашем случае мы ее часто режем ниже)
        text = re.sub(r'[\s,:;\-]+$', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def smart_truncate(text: str, max_length: int, is_collection: bool = False) -> str:
        text = DataTransformer.clean_punctuation(text)

        if len(text) > max_length:
            if is_collection:
                sub_text = text[:max_length]
                last_punct = max(sub_text.rfind('.'), sub_text.rfind('!'), sub_text.rfind('?'))
                if last_punct > 15: text = sub_text[:last_punct+1]
                else: text = sub_text.rsplit(' ', 1)[0]
            else:
                text = text[:max_length].rsplit(' ', 1)[0]

        junk_pattern = r'\s+(и|в|на|с|от|до|за|по|к|из|у|без|для|про|а|но|да|или|как|что|где|когда|если)$'
        for _ in range(3):
            old_text = text
            text = re.sub(junk_pattern, '', text, flags=re.IGNORECASE).strip()
            if text == old_text: break

        text = DataTransformer.clean_punctuation(text)
        if text and is_collection: text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def generate_numeric_id(text: str) -> str:
        return str(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16))[:90]

    @staticmethod
    def clean_emojis_and_specials(text: str) -> str:
        clean_text = re.sub(r'[^\w\s.,!?\-:;/"\'()]', '', str(text))
        return re.sub(r'\s+', ' ', clean_text).strip()

    @staticmethod
    def compress_commercial_text(text: str, max_length: int = 175, is_collection: bool = False) -> str:
        if not text: return ""
        text = re.sub(r'(?i)id\s*товара\s*\d+.*$', '', text)

        fluff_patterns = [
            r'(?i)комментарий от продавца[:\-\s]*', r'(?i)внимание[:\-\s]*', r'(?i)уважаемые покупатели[:\-\s]*',
            r'(?i)мы предоставляем полный пакет документов.*?учёт[:\-\s]*', r'(?i)копия грузовой.*?деклараци[ии].*?(?=\.|\n|$)',
            r'(?i)договор купли-продажи.*?(?=\.|\n|$)', r'(?i)есть аукционный лист.*?(?=\.|\n|$)',
            r'(?i)предоставим подробное фото.*?видео.*?(?=\.|\n|$)', r'(?i)возможна проверка эндоскопом.*?(?=\.|\n|$)',
            r'(?i)цена указана за.*?фото.*?(?=\.|\n|$)', r'(?i)описание товара[:\-\s]*',
            r'(?i)возможна продажа без навесного.*?([.\n]|$)', r'(?i)возможна продажа.*?([.\n]|$)',
            r'(?i)(Номер по производителю|Производитель|Марка|Модель|Год|Кузов|Артикул)[\s:]*$'
        ]
        for pattern in fluff_patterns:
            text = re.sub(pattern, '', text)

        text = re.sub(r':\s*-\s*', ': ', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        
        return DataTransformer.smart_truncate(text, max_length, is_collection)

    @staticmethod
    def parse_universal_price(raw_price_val: Any) -> float:
        if not raw_price_val: return 0.0
        if isinstance(raw_price_val, (int, float)): return float(raw_price_val)
        clean_str = re.sub(r'[^\d.,]', '', str(raw_price_val).replace('\xa0', '').replace(' ', '')).replace(',', '.')
        parts = clean_str.split('.')
        if len(parts) > 2: clean_str = ''.join(parts[:-1]) + '.' + parts[-1]
        try: return float(clean_str) if clean_str else 0.0
        except ValueError: return 0.0

    def apply_title_prefix(self, title: str) -> str:
        prefix = self.config.get("title_prefix", "").strip()
        if not prefix: return title
        words = title.split()
        if not words: return title
        first_word = words[0]
        if not (re.search(r'[A-Za-z0-9]', first_word) or first_word.isupper()):
            words[0] = first_word.lower()
        return f"{prefix} {' '.join(words)}"

    def apply_spin_template(self, tmpl: str, base_text: str) -> str:
        if not base_text: return tmpl.replace("{name}", "")
        words = base_text.split()
        first_word = words[0]
        if not (re.match(r'^[A-Za-z0-9]+$', first_word) or first_word.isupper()):
            words[0] = first_word.lower()
        lowered_text = " ".join(words)

        if tmpl.startswith("{name}"):
            return tmpl.replace("{name}", base_text, 1).replace("{name}", lowered_text)
        else:
            return tmpl.replace("{name}", lowered_text)

    def transform_multiple(self, raw: RawExtractedProduct, url: str, category_id_map: dict) -> list[TransformedProduct]:
        valid_price = self.parse_universal_price(raw.price_raw)
        valid_oldprice = self.parse_universal_price(raw.oldprice_raw)
        if self.config.get("auto_oldprice", True) and valid_oldprice == 0 and valid_price > 0:
            valid_oldprice = valid_price * 1.10

        clean_h1 = self.clean_emojis_and_specials(raw.h1_title)
        words = clean_h1.split()
        type_prefix = words[0] if words else "Товар"

        vendor_clean = self.clean_emojis_and_specials(raw.brand)[:50]
        if vendor_clean.lower() == "unknown" or not vendor_clean:
            vendor_clean = "Noname"

        model_str = clean_h1
        model_str = re.sub(rf'^{re.escape(type_prefix)}\s*', '', model_str, flags=re.IGNORECASE)
        model_str = re.sub(rf'{re.escape(vendor_clean)}\s*', '', model_str, flags=re.IGNORECASE).strip()
        if not model_str: model_str = raw.specs.get('Модель', 'Без модели')

        base_desc_full = self.compress_commercial_text(self.clean_emojis_and_specials(raw.description_usp), max_length=9999, is_collection=False)
        base_desc = self.smart_truncate(base_desc_full, 175, is_collection=False)
        leftover_desc = base_desc_full[len(base_desc):].strip()

        default_sales = self.config.get("default_sales_notes", "").strip()
        extracted_sales = self.clean_emojis_and_specials(raw.sales_notes).strip()

        dynamic_fallback_sales = ""
        if leftover_desc: dynamic_fallback_sales = self.smart_truncate(leftover_desc, 50, is_collection=False)
        if not dynamic_fallback_sales or len(dynamic_fallback_sales) < 5:
            valid_labels = [lbl for lbl in raw.custom_labels if lbl]
            if valid_labels: dynamic_fallback_sales = self.smart_truncate(", ".join(valid_labels), 50, is_collection=False)
        if not dynamic_fallback_sales: dynamic_fallback_sales = "Товар проверен, в наличии"

        safe_specs = {}
        stop_param_keys = ['производитель', 'бренд', 'марка', 'модель']
        stop_param_values = ['none', 'null', 'n/a', 'не указан', 'нет', '-', '', 'стандартный']

        for k, v in raw.specs.items():
            clean_k = self.clean_emojis_and_specials(str(k)).strip()
            clean_v = self.clean_emojis_and_specials(str(v)).strip()
            if clean_k.lower() not in stop_param_keys and clean_v.lower() not in stop_param_values:
                safe_specs[clean_k] = clean_v

        cat_name_clean = self.clean_emojis_and_specials(raw.category_name)[:56]
        cat_id = category_id_map.get(cat_name_clean, self.generate_numeric_id(cat_name_clean)[:10])

        price_str = "0" if valid_price == 0 else f"{valid_price:.2f}"
        oldprice_str = "0" if valid_oldprice == 0 else f"{valid_oldprice:.2f}"

        spin_enabled = self.config.get("spin_enabled", False)
        spin_templates = self.config.get("spin_templates", ["{name}"])
        def_desc_tmpl = self.config.get("default_offer_description", "").strip()

        results = []

        if spin_enabled and spin_templates:
            for i, tmpl in enumerate(spin_templates):
                clean_ai_title = clean_h1
                var_desc = ""
                
                if raw.variations and i < len(raw.variations):
                    var_item = raw.variations[i]
                    if isinstance(var_item, dict):
                        clean_ai_title = self.clean_emojis_and_specials(var_item.get("title", clean_h1))
                        var_desc = var_item.get("description", "")
                    else:
                        clean_ai_title = self.clean_emojis_and_specials(str(var_item))

                clean_ai_title = re.sub(r'(?i)\b(купить|заказать|доставка|в наличии|контрактный|оригинальный)\b\s*', '', clean_ai_title).strip()
                v_title = self.smart_truncate(self.apply_spin_template(tmpl, clean_ai_title), 56, is_collection=False)
                
                if var_desc: 
                    v_desc = self.smart_truncate(self.clean_emojis_and_specials(var_desc), 175, is_collection=False)
                elif def_desc_tmpl: 
                    merged = def_desc_tmpl.replace("{base_desc}", base_desc)
                    v_desc = self.smart_truncate(self.apply_spin_template(merged, clean_ai_title), 175, is_collection=False)
                elif base_desc: 
                    v_desc = self.smart_truncate(base_desc, 175, is_collection=False)
                else: 
                    v_desc = v_title

                if default_sales: final_sales = self.smart_truncate(self.apply_spin_template(default_sales, clean_h1), 50, is_collection=False)
                elif extracted_sales: final_sales = self.smart_truncate(extracted_sales, 50, is_collection=False)
                else: final_sales = dynamic_fallback_sales

                results.append(TransformedProduct(
                    offer_id=self.generate_numeric_id(url + str(i)), url=url, name=v_title, type_prefix=type_prefix, vendor=vendor_clean, model_name=model_str,
                    price=price_str, oldprice=oldprice_str if valid_oldprice > valid_price else "", currency=raw.currency, images=raw.images[:5], description=v_desc, sales_notes=final_sales, specs=safe_specs,
                    custom_labels=[self.clean_emojis_and_specials(lbl)[:175] for lbl in raw.custom_labels[:5]], available="true" if raw.available else "false", category_id=cat_id
                ))
        else:
            prefixed_name = self.apply_title_prefix(clean_h1)
            v_title = self.smart_truncate(prefixed_name, 56, is_collection=False)

            if def_desc_tmpl:
                merged = def_desc_tmpl.replace("{base_desc}", base_desc)
                v_desc = self.smart_truncate(self.apply_spin_template(merged, clean_h1), 175, is_collection=False)
            elif base_desc: v_desc = self.smart_truncate(base_desc, 175, is_collection=False)
            else: v_desc = v_title

            if default_sales: final_sales = self.smart_truncate(self.apply_spin_template(default_sales, clean_h1), 50, is_collection=False)
            elif extracted_sales: final_sales = self.smart_truncate(extracted_sales, 50, is_collection=False)
            else: final_sales = dynamic_fallback_sales

            results.append(TransformedProduct(
                offer_id=self.generate_numeric_id(url), url=url, name=v_title, type_prefix=type_prefix, vendor=vendor_clean, model_name=model_str,
                price=price_str, oldprice=oldprice_str if valid_oldprice > valid_price else "", currency=raw.currency, images=raw.images[:5], description=v_desc, sales_notes=final_sales, specs=safe_specs,
                custom_labels=[self.clean_emojis_and_specials(lbl)[:175] for lbl in raw.custom_labels[:5]], available="true" if raw.available else "false", category_id=cat_id
            ))

        return results

# ==========================================
# 6. ФАЗА СЕРИАЛИЗАЦИИ YML
# ==========================================


class YMLBuilder:
    def __init__(self, config: dict, date_str: str):
        self.config, self.date_str = config, date_str

    def _add_element(self, parent, tag, text, is_desc=False):
        """Безопасное добавление элемента. Если text пустой, элемент не создается."""
        if not text: return
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(text)).strip()
        if not text: return
        
        el = etree.SubElement(parent, tag)
        if self.config.get("cdata_mode", "auto") == "all" or (is_desc and self.config.get("cdata_mode", "auto") == "auto" and re.search(r'[&<>\'"]', text)): 
            el.text = etree.CDATA(text)
        else: 
            el.text = text.replace('"', '&quot;').replace("'", '&apos;')

    def build_feed(self, products: list[TransformedProduct], collections: list[CategoryCollection], output_path: str):
        root = etree.Element('yml_catalog', date=self.date_str)
        shop = etree.SubElement(root, 'shop')
        self._add_element(shop, 'name', self.config.get("shop_name", "Shop"))
        self._add_element(shop, 'company', self.config.get("company_name", "Company"))
        
        url_text = self.config.get("site_url", "https://example.com")
        if url_text: etree.SubElement(shop, 'url').text = url_text

        currencies_el = etree.SubElement(shop, 'currencies')
        for c in sorted(list(set(p.currency for p in products) | {"RUB"})): etree.SubElement(currencies_el, 'currency', id=c, rate="1" if c == "RUB" else "CBRF")

        categories_el = etree.SubElement(shop, 'categories')
        for coll in collections: self._add_element(categories_el, 'category', coll.name)
        
        # Добавляем ID к категориям после их создания
        for idx, cat_el in enumerate(categories_el.findall('category')):
            cat_el.set('id', collections[idx].category_id)

        if self.config.get("feed_mode", "1") in ['1', '2'] and products:
            offers_el = etree.SubElement(shop, 'offers')
            for prod in products:
                offer = etree.SubElement(offers_el, 'offer', id=prod.offer_id, available=prod.available, type="vendor.model")
                self._add_element(offer, 'name', prod.name)
                self._add_element(offer, 'url', prod.url)
                etree.SubElement(offer, 'price').text = prod.price
                if prod.oldprice and prod.oldprice != "0": etree.SubElement(offer, 'oldprice').text = prod.oldprice
                etree.SubElement(offer, 'currencyId').text = prod.currency
                etree.SubElement(offer, 'categoryId').text = prod.category_id
                
                for img in prod.images: self._add_element(offer, 'picture', img)
                self._add_element(offer, 'typePrefix', prod.type_prefix)
                self._add_element(offer, 'vendor', prod.vendor)
                self._add_element(offer, 'model', prod.model_name)
                self._add_element(offer, 'description', prod.description, True)
                self._add_element(offer, 'sales_notes', prod.sales_notes)
                
                for i, lbl in enumerate(prod.custom_labels): self._add_element(offer, f'custom_label_{i}', lbl)
                for key, val in prod.specs.items(): 
                    if val:
                        param_el = etree.SubElement(offer, 'param', name=key)
                        param_el.text = val

        if self.config.get("feed_mode", "1") in ['1', '3']:
            collections_el = etree.SubElement(shop, 'collections')
            def_desc = self.config.get("default_collection_description", "").strip()
            for coll in collections:
                cel = etree.SubElement(collections_el, 'collection', id=coll.category_id)
                self._add_element(cel, 'url', coll.url)
                self._add_element(cel, 'name', coll.name)
                c_desc = coll.description if coll.description else def_desc
                if not c_desc: c_desc = coll.name
                
                final_coll_desc = DataTransformer.compress_commercial_text(c_desc.replace("{name}", coll.name), 81, True)
                self._add_element(cel, 'description', final_coll_desc, True)
                self._add_element(cel, 'picture', coll.picture)
            
            if (self.config.get("duplicate_offers", False) and self.config.get("feed_mode", "1") == '1') or self.config.get("feed_mode", "1") == '3':
                for prod in products:
                    cel = etree.SubElement(collections_el, 'collection', id=f"col_{prod.offer_id}")
                    self._add_element(cel, 'url', prod.url)
                    self._add_element(cel, 'name', prod.name)
                    c_desc = prod.description if prod.description else def_desc
                    if not c_desc: c_desc = prod.name
                    
                    final_coll_desc = DataTransformer.compress_commercial_text(c_desc.replace("{name}", prod.name), 81, True)
                    self._add_element(cel, 'description', final_coll_desc, True)
                    if prod.images: self._add_element(cel, 'picture', prod.images[0])

        etree.ElementTree(root).write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

# ==========================================
# ИНТЕРАКТИВНЫЙ РЕДАКТОР ФИДОВ (YML Editor)
# ==========================================


async def validate_image_url(url: str, session: aiohttp.ClientSession) -> bool:
    try:
        async with session.get(url, timeout=5) as resp:
            if resp.status == 200:
                data = await resp.read()
                img = Image.open(BytesIO(data))
                return img.size[0] >= 450 and img.size[1] >= 450
    except: return False
    return False

# ==========================================
# 2. УМНЫЙ КЭШ И ПРАВИЛА ДОМЕНА
# ==========================================

def load_domain_rules(url: str) -> dict:
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    fname = f"rules_{domain}.json"
    if os.path.exists(fname):
        try:
            with open(fname, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
    return {}


def save_domain_rules(url: str, rules: dict):
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    with open(f"rules_{domain}.json", 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=4, ensure_ascii=False)




def apply_single_pattern(url, html_content, raw_product, pattern_obj):
    if not pattern_obj: return raw_product
    if hasattr(pattern_obj, 'clean_description'):
        raw_product.description_usp = pattern_obj.clean_description(raw_product.description_usp)
    if hasattr(pattern_obj, 'get_custom_labels'):
        new_labels = pattern_obj.get_custom_labels(html_content, raw_product.brand, raw_product.category_name)
        labels_set = set(raw_product.custom_labels)
        labels_set.update(new_labels)
        raw_product.custom_labels = list(labels_set)[:5]
    return raw_product

async def run_github_worker():
    with open("feed_settings.json", "r", encoding="utf-8") as f: config = json.load(f)
    t_urls = config.get("target_urls", [])
    if not t_urls: return
    
    print("🚀 Запуск GitHub Worker (Classic-Mode + Вшитый паттерн)...")
    transformer = DataTransformer(config)
    discovery_agent = DiscoveryAgent()
    cache_manager = CacheManager("feed_cache.json")
    scraper = ClassicScraper()
    
    try: active_pattern = Pattern()
    except Exception: active_pattern = None
    
    skip_empty_price = config.get("skip_empty_price", True)
    output_filename = config.get("output_file", "feed.xml")
    
    for base_url in t_urls:
        transformed_products = []
        collections_map = {}
        category_id_map = {}
        domain_rules = load_domain_rules(base_url)
        
        url_queue = [base_url]
        visited_urls = set()
        crawled_product_urls = set()
        semaphore = asyncio.Semaphore(1)
        
        async def process_single_product(product_url: str, parent_category_url: str, crawler_instance):
            async with semaphore:
                try:
                    crawled_product_urls.add(product_url)
                    await asyncio.sleep(2.5)
                    result = await crawler_instance.arun(url=product_url, bypass_cache=True, magic=True, delay_before_return_html=2.0)
                    if not result or not result.html: return
                    
                    is_cached, cached_raw_data = cache_manager.check_cache(product_url, result.html)
                    url_in_cache = product_url in cache_manager.cache
                    
                    if is_cached and cached_raw_data:
                        if "price_raw" in cached_raw_data: cached_raw_data["price"] = cached_raw_data.pop("price_raw")
                        if "oldprice_raw" in cached_raw_data: cached_raw_data["oldprice"] = cached_raw_data.pop("oldprice_raw")
                        raw_product = RawExtractedProduct(**cached_raw_data)
                    elif url_in_cache:
                        temp_raw = scraper.extract_product_data(product_url, result.html, result.markdown, domain_rules)
                        cache_manager.patch_cache(product_url, temp_raw.price_raw, temp_raw.oldprice_raw, temp_raw.available)
                        cached_raw_data = cache_manager.get_raw_data(product_url)
                        if "price_raw" in cached_raw_data: cached_raw_data["price"] = cached_raw_data.pop("price_raw")
                        if "oldprice_raw" in cached_raw_data: cached_raw_data["oldprice"] = cached_raw_data.pop("oldprice_raw")
                        raw_product = RawExtractedProduct(**cached_raw_data)
                        print(f"  [Smart Merge] Обновлен товар: {product_url}")
                    else:
                        raw_product = scraper.extract_product_data(product_url, result.html, result.markdown, domain_rules)
                        raw_product = apply_single_pattern(product_url, result.html, raw_product, active_pattern)

                        semantic_pattern = domain_rules.get("semantic_pattern", "")
                        if semantic_pattern and not raw_product.description_usp:
                            text = semantic_pattern.replace("{brand}", raw_product.brand).replace("{category_name}", raw_product.category_name)
                            for k, v in raw_product.specs.items(): 
                                text = text.replace("{specs[" + str(k) + "]}", str(v))
                            raw_product.description_usp = re.sub(r'\{specs\[.*?\]\}', '', text).strip()
                            
                        cache_manager.update_cache(product_url, result.html, raw_product.model_dump())
                        print(f"  [New] Спарсен новый товар: {product_url}")
                        
                    multi_products = transformer.transform_multiple(raw_product, product_url, category_id_map)
                    for p in multi_products:
                        if p.category_id not in collections_map:
                            c_name = transformer.clean_emojis_and_specials(raw_product.category_name)[:56]
                            c_desc = transformer.smart_truncate(transformer.clean_emojis_and_specials(raw_product.category_usp), 81, is_collection=True)
                            collections_map[p.category_id] = CategoryCollection(category_id=p.category_id, name=c_name, url=parent_category_url, picture=raw_product.images[0] if raw_product.images else "", description=c_desc)
                        if p.price != "0" or not skip_empty_price:
                            transformed_products.append(p)
                except Exception as e: print(f"Error: {e}")

        async with AsyncWebCrawler() as crawler:
            while url_queue:
                current_url = url_queue.pop(0)
                if current_url in visited_urls: continue
                visited_urls.add(current_url)
                
                result = await crawler.arun(url=current_url, bypass_cache=True, magic=True, delay_before_return_html=1.5)
                if not result or not result.html: continue
                
                grouped_links = discovery_agent.analyze_and_group_links(current_url, result.html)
                if not grouped_links:
                    await process_single_product(current_url, current_url, crawler)
                    continue
                    
                chosen_prod_sigs = set(config.get("auto_product_signatures", []))
                chosen_cat_sigs = set(config.get("auto_category_signatures", []))
                
                product_items = []
                for sig, links in grouped_links.items():
                    if sig in chosen_prod_sigs: product_items.extend(links)
                    elif sig in chosen_cat_sigs:
                        for item in links:
                            if item['url'] not in visited_urls and item['url'] not in url_queue: url_queue.append(item['url'])
                
                if product_items:
                    tasks = [process_single_product(item['url'], current_url, crawler) for item in product_items]
                    await asyncio.gather(*tasks)

        all_cached_urls = cache_manager.get_all_cached_urls()
        missing_urls = {u for u in all_cached_urls if u not in crawled_product_urls and u.startswith(base_url)}
        
        for m_url in missing_urls:
            raw_data_dict = cache_manager.get_raw_data(m_url)
            if raw_data_dict:
                raw_product = RawExtractedProduct(**raw_data_dict)
                raw_product.available = False
                cat_name_clean = transformer.clean_emojis_and_specials(raw_product.category_name)[:56]
                cat_id = category_id_map.get(cat_name_clean, transformer.generate_numeric_id(cat_name_clean)[:10])
                transformed_products_archived = transformer.transform_multiple(raw_product, m_url, category_id_map)
                transformed_products.extend(transformed_products_archived)
                
        if transformed_products:
            builder = YMLBuilder(config, datetime.now().strftime("%Y-%m-%d %H:%M"))
            builder.build_feed(transformed_products, list(collections_map.values()), output_filename)
            print(f"🎉 Фид сохранен: {output_filename}")

if __name__ == "__main__":
    asyncio.run(run_github_worker())
