"""
HLP Collection - 24/7 Text Collection Agent with Web Scraping and OCR

This module provides comprehensive text collection capabilities for
open access historical linguistics texts with a focus on Greek
(especially Medieval/Byzantine Greek) and other Indo-European languages.

Features:
- Headless web scraping with Selenium
- OCR integration for digitized manuscripts (Tesseract, EasyOCR)
- Multi-source collection (Perseus, TLG, First1KGreek, Internet Archive)
- Byzantine and Medieval Greek text sources
- Rate limiting and polite crawling
- 24/7 continuous operation

Supported Languages:
- Greek (Ancient, Hellenistic, Byzantine, Modern)
- Latin (Classical, Medieval, Renaissance)
- English (Old, Middle, Early Modern, Modern)
- Gothic, Old Church Slavonic, Sanskrit, Armenian

University of Athens - Nikolaos Lavidas
"""

from hlp_collection.collection_agent import (
    CollectionAgent,
    CollectionConfig,
    CollectionSource,
    CollectionJob,
)

from hlp_collection.web_scraper import (
    WebScraper,
    ScraperConfig,
    HeadlessBrowser,
)

from hlp_collection.ocr_engine import (
    OCREngine,
    TesseractOCR,
    EasyOCREngine,
    OCRResult,
)

from hlp_collection.sources import (
    PerseusCollector,
    First1KGreekCollector,
    InternetArchiveCollector,
    ByzantineTextCollector,
    GutenbergCollector,
)

__all__ = [
    'CollectionAgent',
    'CollectionConfig',
    'CollectionSource',
    'CollectionJob',
    'WebScraper',
    'ScraperConfig',
    'HeadlessBrowser',
    'OCREngine',
    'TesseractOCR',
    'EasyOCREngine',
    'OCRResult',
    'PerseusCollector',
    'First1KGreekCollector',
    'InternetArchiveCollector',
    'ByzantineTextCollector',
    'GutenbergCollector',
]
