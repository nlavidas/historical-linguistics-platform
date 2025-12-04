"""
Web Scraper - Headless web scraping for text collection

This module provides headless web scraping capabilities using Selenium
for collecting open access texts from various digital libraries.

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import time
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    headless: bool = True
    user_agent: str = "NKUA-Historical-Linguistics-Platform/1.0 (Academic Research; +https://github.com/nlavidas/historical-linguistics-platform)"
    timeout: int = 30
    page_load_timeout: int = 60
    implicit_wait: int = 10
    min_delay: float = 2.0
    max_delay: float = 5.0
    max_retries: int = 3
    respect_robots_txt: bool = True
    download_images: bool = False
    screenshot_on_error: bool = True
    log_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'headless': self.headless,
            'user_agent': self.user_agent,
            'timeout': self.timeout,
            'page_load_timeout': self.page_load_timeout,
            'min_delay': self.min_delay,
            'max_delay': self.max_delay,
            'max_retries': self.max_retries,
        }


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str
    html: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'content_length': len(self.content),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'link_count': len(self.links),
            'success': self.success,
            'error_message': self.error_message,
        }


class HeadlessBrowser:
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.driver = None
        self._initialized = False
    
    def initialize(self) -> bool:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            
            options = Options()
            
            if self.config.headless:
                options.add_argument('--headless=new')
            
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument(f'--user-agent={self.config.user_agent}')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-automation'])
            options.add_experimental_option('useAutomationExtension', False)
            
            if not self.config.download_images:
                prefs = {'profile.managed_default_content_settings.images': 2}
                options.add_experimental_option('prefs', prefs)
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(self.config.page_load_timeout)
            self.driver.implicitly_wait(self.config.implicit_wait)
            
            self._initialized = True
            logger.info("Headless browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize headless browser: {e}")
            return False
    
    def get_page(self, url: str) -> Optional[ScrapedPage]:
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            self.driver.get(url)
            
            self._random_delay()
            
            title = self.driver.title
            html = self.driver.page_source
            
            content = self._extract_text_content()
            
            links = self._extract_links()
            images = self._extract_images() if self.config.download_images else []
            
            metadata = self._extract_metadata()
            
            return ScrapedPage(
                url=url,
                title=title,
                content=content,
                html=html,
                timestamp=datetime.now(),
                metadata=metadata,
                links=links,
                images=images,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            
            if self.config.screenshot_on_error and self.config.log_dir:
                self._save_error_screenshot(url)
            
            return ScrapedPage(
                url=url,
                title="",
                content="",
                html="",
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def _extract_text_content(self) -> str:
        try:
            from selenium.webdriver.common.by import By
            
            body = self.driver.find_element(By.TAG_NAME, 'body')
            text = body.text
            
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            return ""
    
    def _extract_links(self) -> List[str]:
        try:
            from selenium.webdriver.common.by import By
            
            links = []
            elements = self.driver.find_elements(By.TAG_NAME, 'a')
            
            for elem in elements:
                href = elem.get_attribute('href')
                if href and href.startswith('http'):
                    links.append(href)
            
            return list(set(links))
            
        except Exception as e:
            logger.error(f"Error extracting links: {e}")
            return []
    
    def _extract_images(self) -> List[str]:
        try:
            from selenium.webdriver.common.by import By
            
            images = []
            elements = self.driver.find_elements(By.TAG_NAME, 'img')
            
            for elem in elements:
                src = elem.get_attribute('src')
                if src:
                    images.append(src)
            
            return list(set(images))
            
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return []
    
    def _extract_metadata(self) -> Dict[str, Any]:
        metadata = {}
        
        try:
            from selenium.webdriver.common.by import By
            
            meta_tags = self.driver.find_elements(By.TAG_NAME, 'meta')
            
            for meta in meta_tags:
                name = meta.get_attribute('name') or meta.get_attribute('property')
                content = meta.get_attribute('content')
                
                if name and content:
                    metadata[name] = content
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _random_delay(self):
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        time.sleep(delay)
    
    def _save_error_screenshot(self, url: str):
        try:
            if self.config.log_dir:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"error_{timestamp}.png"
                filepath = self.config.log_dir / filename
                self.driver.save_screenshot(str(filepath))
                logger.info(f"Error screenshot saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save error screenshot: {e}")
    
    def execute_script(self, script: str) -> Any:
        if not self._initialized:
            return None
        
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return None
    
    def scroll_to_bottom(self):
        if not self._initialized:
            return
        
        try:
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                
        except Exception as e:
            logger.error(f"Error scrolling: {e}")
    
    def wait_for_element(self, selector: str, by: str = "css", timeout: int = 10) -> bool:
        if not self._initialized:
            return False
        
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
                'tag': By.TAG_NAME,
            }
            
            by_type = by_map.get(by, By.CSS_SELECTOR)
            
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by_type, selector))
            )
            return True
            
        except Exception:
            return False
    
    def click_element(self, selector: str, by: str = "css") -> bool:
        if not self._initialized:
            return False
        
        try:
            from selenium.webdriver.common.by import By
            
            by_map = {
                'css': By.CSS_SELECTOR,
                'xpath': By.XPATH,
                'id': By.ID,
                'class': By.CLASS_NAME,
            }
            
            by_type = by_map.get(by, By.CSS_SELECTOR)
            element = self.driver.find_element(by_type, selector)
            element.click()
            return True
            
        except Exception as e:
            logger.error(f"Error clicking element: {e}")
            return False
    
    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            self._initialized = False
            logger.info("Headless browser closed")


class WebScraper:
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.browser = HeadlessBrowser(self.config)
        self.visited_urls: set = set()
        self.failed_urls: set = set()
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'total_content_bytes': 0,
            'start_time': None,
        }
    
    def scrape_url(self, url: str) -> Optional[ScrapedPage]:
        if url in self.visited_urls:
            logger.debug(f"URL already visited: {url}")
            return None
        
        if url in self.failed_urls:
            logger.debug(f"URL previously failed: {url}")
            return None
        
        self.visited_urls.add(url)
        
        for attempt in range(self.config.max_retries):
            page = self.browser.get_page(url)
            
            if page and page.success:
                self.stats['pages_scraped'] += 1
                self.stats['total_content_bytes'] += len(page.content)
                return page
            
            if attempt < self.config.max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.warning(f"Retry {attempt + 1} for {url} in {wait_time}s")
                time.sleep(wait_time)
        
        self.failed_urls.add(url)
        self.stats['pages_failed'] += 1
        return page
    
    def scrape_urls(self, urls: List[str]) -> List[ScrapedPage]:
        results = []
        
        self.stats['start_time'] = datetime.now()
        
        for url in urls:
            page = self.scrape_url(url)
            if page:
                results.append(page)
        
        return results
    
    def crawl_site(
        self,
        start_url: str,
        max_pages: int = 100,
        url_pattern: Optional[str] = None,
        same_domain_only: bool = True
    ) -> List[ScrapedPage]:
        results = []
        to_visit = [start_url]
        
        parsed_start = urlparse(start_url)
        start_domain = parsed_start.netloc
        
        pattern = re.compile(url_pattern) if url_pattern else None
        
        self.stats['start_time'] = datetime.now()
        
        while to_visit and len(results) < max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            page = self.scrape_url(url)
            
            if page and page.success:
                results.append(page)
                
                for link in page.links:
                    if link in self.visited_urls:
                        continue
                    
                    parsed_link = urlparse(link)
                    
                    if same_domain_only and parsed_link.netloc != start_domain:
                        continue
                    
                    if pattern and not pattern.match(link):
                        continue
                    
                    to_visit.append(link)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats['visited_count'] = len(self.visited_urls)
        stats['failed_count'] = len(self.failed_urls)
        
        if stats['start_time']:
            elapsed = (datetime.now() - stats['start_time']).total_seconds()
            stats['elapsed_seconds'] = elapsed
            stats['pages_per_minute'] = (stats['pages_scraped'] / elapsed * 60) if elapsed > 0 else 0
        
        return stats
    
    def close(self):
        self.browser.close()
