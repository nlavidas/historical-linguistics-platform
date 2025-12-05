"""
OCR Engine - Optical Character Recognition for digitized manuscripts

This module provides OCR capabilities for extracting text from
digitized manuscripts and historical documents.

Supports:
- Tesseract OCR (with Greek, Latin, Old English models)
- EasyOCR (deep learning based)
- Custom preprocessing for historical documents

University of Athens - Nikolaos Lavidas
"""

from __future__ import annotations
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import io

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    language: str
    source_file: Optional[str] = None
    processing_time_ms: float = 0
    word_count: int = 0
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'source_file': self.source_file,
            'processing_time_ms': self.processing_time_ms,
            'word_count': self.word_count,
            'metadata': self.metadata,
        }


@dataclass
class OCRConfig:
    language: str = "eng"
    dpi: int = 300
    psm: int = 3
    oem: int = 3
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = True
    binarize: bool = True
    confidence_threshold: float = 60.0
    
    LANGUAGE_CODES = {
        'english': 'eng',
        'greek': 'grc+ell',
        'ancient_greek': 'grc',
        'modern_greek': 'ell',
        'latin': 'lat',
        'german': 'deu',
        'french': 'fra',
        'italian': 'ita',
        'spanish': 'spa',
        'russian': 'rus',
        'old_english': 'ang',
        'gothic': 'got',
        'armenian': 'hye',
        'georgian': 'kat',
        'coptic': 'cop',
        'syriac': 'syr',
        'hebrew': 'heb',
        'arabic': 'ara',
        'sanskrit': 'san',
    }
    
    @classmethod
    def for_language(cls, language: str) -> OCRConfig:
        lang_code = cls.LANGUAGE_CODES.get(language.lower(), language)
        return cls(language=lang_code)
    
    @classmethod
    def for_greek(cls) -> OCRConfig:
        return cls(language='grc+ell', preprocessing=True, deskew=True)
    
    @classmethod
    def for_byzantine_greek(cls) -> OCRConfig:
        return cls(language='grc+ell', preprocessing=True, deskew=True, denoise=True)
    
    @classmethod
    def for_latin(cls) -> OCRConfig:
        return cls(language='lat', preprocessing=True)


class OCREngine(ABC):
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def process_image(self, image_path: str) -> OCRResult:
        pass
    
    @abstractmethod
    def process_image_bytes(self, image_bytes: bytes) -> OCRResult:
        pass
    
    def process_pdf(self, pdf_path: str) -> List[OCRResult]:
        results = []
        
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                pix = page.get_pixmap(dpi=self.config.dpi)
                img_bytes = pix.tobytes("png")
                
                result = self.process_image_bytes(img_bytes)
                result.metadata['page_number'] = page_num + 1
                result.metadata['total_pages'] = len(doc)
                result.source_file = f"{pdf_path}#page={page_num + 1}"
                
                results.append(result)
            
            doc.close()
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return results
    
    def process_url(self, url: str) -> OCRResult:
        try:
            import requests
            
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                return self.process_image_bytes(response.content)
            else:
                return OCRResult(
                    text="",
                    confidence=0,
                    language=self.config.language,
                    metadata={'error': f"HTTP {response.status_code}"}
                )
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return OCRResult(
                text="",
                confidence=0,
                language=self.config.language,
                metadata={'error': str(e)}
            )
    
    def _preprocess_image(self, image):
        try:
            import cv2
            import numpy as np
            
            if isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(str(image))
            
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.config.denoise:
                gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            if self.config.binarize:
                gray = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
            
            if self.config.deskew:
                coords = np.column_stack(np.where(gray > 0))
                if len(coords) > 0:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = 90 + angle
                    if abs(angle) > 0.5:
                        (h, w) = gray.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        gray = cv2.warpAffine(
                            gray, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
            
            return gray
            
        except ImportError:
            logger.warning("OpenCV not installed. Skipping preprocessing.")
            return None
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None


class TesseractOCR(OCREngine):
    
    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)
        self.tesseract = None
    
    def initialize(self) -> bool:
        try:
            import pytesseract
            
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized (version {version})")
            
            self.tesseract = pytesseract
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            return False
    
    def process_image(self, image_path: str) -> OCRResult:
        if not self._initialized:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0,
                    language=self.config.language,
                    metadata={'error': 'Tesseract not initialized'}
                )
        
        import time
        start_time = time.time()
        
        try:
            from PIL import Image
            
            if self.config.preprocessing:
                preprocessed = self._preprocess_image(image_path)
                if preprocessed is not None:
                    import cv2
                    _, buffer = cv2.imencode('.png', preprocessed)
                    img = Image.open(io.BytesIO(buffer.tobytes()))
                else:
                    img = Image.open(image_path)
            else:
                img = Image.open(image_path)
            
            custom_config = f'--oem {self.config.oem} --psm {self.config.psm}'
            
            text = self.tesseract.image_to_string(
                img,
                lang=self.config.language,
                config=custom_config
            )
            
            data = self.tesseract.image_to_data(
                img,
                lang=self.config.language,
                config=custom_config,
                output_type=self.tesseract.Output.DICT
            )
            
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            bounding_boxes = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    bounding_boxes.append({
                        'text': data['text'][i],
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': data['conf'][i],
                    })
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                language=self.config.language,
                source_file=image_path,
                processing_time_ms=processing_time,
                bounding_boxes=bounding_boxes,
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0,
                language=self.config.language,
                source_file=image_path,
                metadata={'error': str(e)}
            )
    
    def process_image_bytes(self, image_bytes: bytes) -> OCRResult:
        if not self._initialized:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0,
                    language=self.config.language,
                    metadata={'error': 'Tesseract not initialized'}
                )
        
        import time
        start_time = time.time()
        
        try:
            from PIL import Image
            
            if self.config.preprocessing:
                preprocessed = self._preprocess_image(image_bytes)
                if preprocessed is not None:
                    import cv2
                    _, buffer = cv2.imencode('.png', preprocessed)
                    img = Image.open(io.BytesIO(buffer.tobytes()))
                else:
                    img = Image.open(io.BytesIO(image_bytes))
            else:
                img = Image.open(io.BytesIO(image_bytes))
            
            custom_config = f'--oem {self.config.oem} --psm {self.config.psm}'
            
            text = self.tesseract.image_to_string(
                img,
                lang=self.config.language,
                config=custom_config
            )
            
            data = self.tesseract.image_to_data(
                img,
                lang=self.config.language,
                config=custom_config,
                output_type=self.tesseract.Output.DICT
            )
            
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                language=self.config.language,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return OCRResult(
                text="",
                confidence=0,
                language=self.config.language,
                metadata={'error': str(e)}
            )


class EasyOCREngine(OCREngine):
    
    def __init__(self, config: Optional[OCRConfig] = None):
        super().__init__(config)
        self.reader = None
    
    def initialize(self) -> bool:
        try:
            import easyocr
            
            lang_map = {
                'eng': ['en'],
                'grc': ['el'],
                'ell': ['el'],
                'grc+ell': ['el'],
                'lat': ['la'],
                'deu': ['de'],
                'fra': ['fr'],
                'ita': ['it'],
                'spa': ['es'],
                'rus': ['ru'],
            }
            
            languages = lang_map.get(self.config.language, ['en'])
            
            self.reader = easyocr.Reader(languages, gpu=False)
            
            logger.info(f"EasyOCR initialized for languages: {languages}")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return False
    
    def process_image(self, image_path: str) -> OCRResult:
        if not self._initialized:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0,
                    language=self.config.language,
                    metadata={'error': 'EasyOCR not initialized'}
                )
        
        import time
        start_time = time.time()
        
        try:
            if self.config.preprocessing:
                preprocessed = self._preprocess_image(image_path)
                if preprocessed is not None:
                    results = self.reader.readtext(preprocessed)
                else:
                    results = self.reader.readtext(image_path)
            else:
                results = self.reader.readtext(image_path)
            
            texts = []
            confidences = []
            bounding_boxes = []
            
            for (bbox, text, conf) in results:
                texts.append(text)
                confidences.append(conf)
                bounding_boxes.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf,
                })
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=self.config.language,
                source_file=image_path,
                processing_time_ms=processing_time,
                bounding_boxes=bounding_boxes,
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0,
                language=self.config.language,
                source_file=image_path,
                metadata={'error': str(e)}
            )
    
    def process_image_bytes(self, image_bytes: bytes) -> OCRResult:
        if not self._initialized:
            if not self.initialize():
                return OCRResult(
                    text="",
                    confidence=0,
                    language=self.config.language,
                    metadata={'error': 'EasyOCR not initialized'}
                )
        
        import time
        start_time = time.time()
        
        try:
            import numpy as np
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            if self.config.preprocessing:
                preprocessed = self._preprocess_image(image_bytes)
                if preprocessed is not None:
                    results = self.reader.readtext(preprocessed)
                else:
                    import cv2
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    results = self.reader.readtext(img)
            else:
                import cv2
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                results = self.reader.readtext(img)
            
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                texts.append(text)
                confidences.append(conf)
            
            full_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=self.config.language,
                processing_time_ms=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return OCRResult(
                text="",
                confidence=0,
                language=self.config.language,
                metadata={'error': str(e)}
            )


class OCRFactory:
    
    @staticmethod
    def create(engine_type: str = "tesseract", config: Optional[OCRConfig] = None) -> OCREngine:
        engines = {
            'tesseract': TesseractOCR,
            'easyocr': EasyOCREngine,
        }
        
        engine_class = engines.get(engine_type.lower(), TesseractOCR)
        return engine_class(config)
    
    @staticmethod
    def create_for_greek() -> OCREngine:
        config = OCRConfig.for_greek()
        return TesseractOCR(config)
    
    @staticmethod
    def create_for_byzantine() -> OCREngine:
        config = OCRConfig.for_byzantine_greek()
        return TesseractOCR(config)
    
    @staticmethod
    def create_for_latin() -> OCREngine:
        config = OCRConfig.for_latin()
        return TesseractOCR(config)
