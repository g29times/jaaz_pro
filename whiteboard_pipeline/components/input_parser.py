import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import logging

from ..interfaces import InputParserInterface
from ..models import WhiteboardInput, ParsedInput, ParsedElement, InputType


class InputParser(InputParserInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # OCR is now mandatory
        self.ocr_engine = None
        self.ocr_confidence_threshold = self.config.get('ocr_confidence_threshold', 0.3)  # Lower threshold for better recall
        self.backup_ocr_engine = None
        
        # Enhanced logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info("Initializing InputParser with mandatory OCR")
        
        self._initialize_ocr_engines()
        
        # Visual element classification parameters
        self.element_types = {
            'text': 0.8,
            'arrow': 0.7, 
            'sketch': 0.6,
            'diagram': 0.7,
            'table': 0.5,
            'flowchart': 0.8  # New focus on flowchart elements
        }
    
    def _initialize_ocr_engines(self):
        """Initialize OCR engines with fallback options"""
        
        # Primary OCR: PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                show_log=False,
                use_gpu=True,  # Enable GPU if available
                det_db_thresh=0.3,  # Lower detection threshold
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6
            )
            self.logger.info("PaddleOCR initialized successfully as primary OCR engine")
        except ImportError as e:
            self.logger.error(f"PaddleOCR not available: {e}")
            self.ocr_engine = None
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.ocr_engine = None
        
        # Backup OCR: EasyOCR
        try:
            import easyocr
            self.backup_ocr_engine = easyocr.Reader(['en'], gpu=True)
            self.logger.info("EasyOCR initialized successfully as backup OCR engine")
        except ImportError:
            self.logger.warning("EasyOCR not available as backup")
            self.backup_ocr_engine = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize EasyOCR backup: {e}")
            self.backup_ocr_engine = None
        
        # Check if we have at least one OCR engine
        if self.ocr_engine is None and self.backup_ocr_engine is None:
            raise RuntimeError(
                "No OCR engines available. Please install PaddleOCR or EasyOCR:\n"
                "pip install paddleocr\n"
                "pip install easyocr"
            )
    
    async def parse(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse whiteboard input with mandatory OCR"""
        self.logger.info(f"Parsing input of type: {whiteboard_input.input_type}")
        
        # Log input metadata for feedback
        if whiteboard_input.metadata:
            self.logger.info(f"Input metadata: {whiteboard_input.metadata}")
        
        if whiteboard_input.input_type == InputType.TEXT:
            return await self._parse_text(whiteboard_input)
        elif whiteboard_input.input_type in [InputType.SKETCH, InputType.IMAGE]:
            return await self._parse_image(whiteboard_input)
        elif whiteboard_input.input_type == InputType.PDF:
            return await self._parse_pdf(whiteboard_input)
        else:
            raise ValueError(f"Unsupported input type: {whiteboard_input.input_type}")
    
    async def _parse_text(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse text input with enhanced processing"""
        text_content = str(whiteboard_input.content)
        
        self.logger.info(f"Processing text input, length: {len(text_content)}")
        
        # Enhanced text analysis for flowchart detection
        flowchart_keywords = self._detect_flowchart_keywords(text_content)
        
        elements = [
            ParsedElement(
                element_type="text",
                content=text_content,
                confidence=1.0,
                metadata={
                    "source": "direct_text",
                    "flowchart_keywords": flowchart_keywords,
                    "word_count": len(text_content.split()),
                    "character_count": len(text_content)
                }
            )
        ]
        
        self.logger.info(f"Found {len(flowchart_keywords)} flowchart keywords: {flowchart_keywords}")
        
        return ParsedInput(
            elements=elements,
            raw_text=text_content,
            metadata={
                "input_type": "text", 
                "length": len(text_content),
                "flowchart_indicators": len(flowchart_keywords),
                "processing_method": "direct_text"
            }
        )
    
    def _detect_flowchart_keywords(self, text: str) -> List[str]:
        """Detect keywords that indicate flowchart content"""
        flowchart_keywords = [
            'flow', 'process', 'step', 'workflow', 'diagram', 'flowchart',
            'start', 'end', 'begin', 'finish', 'decision', 'if', 'then', 'else',
            'arrow', 'next', 'previous', 'sequence', 'procedure', 'algorithm',
            'branch', 'loop', 'while', 'for', 'until', 'condition'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in flowchart_keywords if kw in text_lower]
        
        return found_keywords
    
    async def _parse_image(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse image input with mandatory OCR"""
        image_data = self._prepare_image_data(whiteboard_input)
        
        self.logger.info(f"Processing image input, size: {len(image_data)} bytes")
        
        # Mandatory OCR processing
        ocr_results = await self._perform_mandatory_ocr(image_data)
        
        # Visual element classification
        visual_elements = await self._classify_visual_elements(image_data)
        
        all_elements = ocr_results + visual_elements
        raw_text = " ".join([elem.content for elem in ocr_results])
        
        # Enhanced logging for feedback
        self.logger.info(f"OCR extracted {len(ocr_results)} text elements")
        self.logger.info(f"Visual classification found {len(visual_elements)} elements")
        self.logger.info(f"Total raw text length: {len(raw_text)}")
        
        # Log confidence distribution
        confidences = [elem.confidence for elem in all_elements]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            self.logger.info(f"Average element confidence: {avg_confidence:.3f}")
        
        return ParsedInput(
            elements=all_elements,
            raw_text=raw_text,
            metadata={
                "input_type": "image",
                "image_size_bytes": len(image_data),
                "ocr_elements": len(ocr_results),
                "visual_elements": len(visual_elements),
                "total_elements": len(all_elements),
                "processing_method": "ocr_mandatory"
            }
        )
    
    async def _perform_mandatory_ocr(self, image_data: bytes) -> List[ParsedElement]:
        """Perform OCR with primary and backup engines"""
        
        # Try primary OCR engine first
        if self.ocr_engine:
            try:
                self.logger.debug("Attempting OCR with PaddleOCR")
                results = await self._ocr_with_paddleocr(image_data)
                if results:
                    self.logger.info(f"PaddleOCR extracted {len(results)} text elements")
                    return results
                else:
                    self.logger.warning("PaddleOCR returned no results")
            except Exception as e:
                self.logger.error(f"PaddleOCR failed: {e}")
        
        # Try backup OCR engine
        if self.backup_ocr_engine:
            try:
                self.logger.debug("Attempting OCR with EasyOCR backup")
                results = await self._ocr_with_easyocr(image_data)
                if results:
                    self.logger.info(f"EasyOCR backup extracted {len(results)} text elements")
                    return results
                else:
                    self.logger.warning("EasyOCR backup returned no results")
            except Exception as e:
                self.logger.error(f"EasyOCR backup failed: {e}")
        
        # If both OCR engines fail, this is a critical error since OCR is mandatory
        self.logger.error("All OCR engines failed - this violates the mandatory OCR requirement")
        raise RuntimeError(
            "Mandatory OCR failed: No text could be extracted from the image. "
            "Check OCR engine configuration and image quality."
        )
    
    async def _ocr_with_paddleocr(self, image_data: bytes) -> List[ParsedElement]:
        """Perform OCR using PaddleOCR"""
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        results = self.ocr_engine.ocr(image_array, cls=True)
        elements = []
        
        for line in results[0] or []:
            bbox, (text, confidence) = line
            
            if confidence >= self.ocr_confidence_threshold:
                elements.append(
                    ParsedElement(
                        element_type="text",
                        content=text.strip(),
                        confidence=confidence,
                        bbox=[int(coord) for point in bbox for coord in point],
                        metadata={
                            "source": "paddleocr",
                            "original_confidence": confidence,
                            "bbox_area": self._calculate_bbox_area(bbox)
                        }
                    )
                )
                self.logger.debug(f"OCR found: '{text.strip()}' (confidence: {confidence:.3f})")
            else:
                self.logger.debug(f"OCR rejected low confidence text: '{text}' (confidence: {confidence:.3f})")
        
        return elements
    
    async def _ocr_with_easyocr(self, image_data: bytes) -> List[ParsedElement]:
        """Perform OCR using EasyOCR backup"""
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        results = self.backup_ocr_engine.readtext(image_array)
        elements = []
        
        for (bbox, text, confidence) in results:
            if confidence >= self.ocr_confidence_threshold:
                elements.append(
                    ParsedElement(
                        element_type="text",
                        content=text.strip(),
                        confidence=confidence,
                        bbox=[int(coord) for point in bbox for coord in point],
                        metadata={
                            "source": "easyocr_backup",
                            "original_confidence": confidence,
                            "bbox_area": self._calculate_bbox_area(bbox)
                        }
                    )
                )
                self.logger.debug(f"Backup OCR found: '{text.strip()}' (confidence: {confidence:.3f})")
            else:
                self.logger.debug(f"Backup OCR rejected low confidence text: '{text}' (confidence: {confidence:.3f})")
        
        return elements
    
    def _calculate_bbox_area(self, bbox) -> float:
        """Calculate bounding box area for metadata"""
        try:
            if len(bbox) == 4:  # 4 points
                points = np.array(bbox)
                area = cv2.contourArea(points)
                return float(area)
            return 0.0
        except Exception:
            return 0.0
    
    async def _parse_pdf(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        try:
            import fitz  # PyMuPDF
            
            if isinstance(whiteboard_input.content, Path):
                pdf_path = whiteboard_input.content
            else:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(whiteboard_input.content)
                    pdf_path = Path(tmp.name)
            
            doc = fitz.open(pdf_path)
            all_elements = []
            raw_text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                text = page.get_text()
                raw_text_parts.append(text)
                
                if text.strip():
                    all_elements.append(
                        ParsedElement(
                            element_type="text",
                            content=text,
                            confidence=0.9,
                            metadata={"page": page_num, "source": "pdf_text"}
                        )
                    )
                
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                visual_elements = await self._classify_visual_elements(img_data)
                all_elements.extend(visual_elements)
            
            doc.close()
            
            return ParsedInput(
                elements=all_elements,
                raw_text=" ".join(raw_text_parts),
                metadata={"input_type": "pdf", "pages": len(doc)}
            )
            
        except ImportError:
            self.logger.error("PyMuPDF not available for PDF processing")
            raise
    
    def _prepare_image_data(self, whiteboard_input: WhiteboardInput) -> bytes:
        if isinstance(whiteboard_input.content, bytes):
            return whiteboard_input.content
        elif isinstance(whiteboard_input.content, Path):
            return whiteboard_input.content.read_bytes()
        else:
            raise ValueError("Invalid image data format")
    
    async def _perform_ocr(self, image_data: bytes) -> List[ParsedElement]:
        if not self.ocr_engine:
            return self._mock_ocr_results()
        
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            results = self.ocr_engine.ocr(image_array, cls=True)
            elements = []
            
            for line in results[0] or []:
                bbox, (text, confidence) = line
                if confidence > 0.5:
                    elements.append(
                        ParsedElement(
                            element_type="text",
                            content=text,
                            confidence=confidence,
                            bbox=[int(coord) for point in bbox for coord in point],
                            metadata={"source": "ocr"}
                        )
                    )
            
            return elements
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return []
    
    async def _classify_visual_elements(self, image_data: bytes) -> List[ParsedElement]:
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            elements = []
            
            if self._detect_arrows(image_array):
                elements.append(
                    ParsedElement(
                        element_type="arrow",
                        content="directional_arrow",
                        confidence=0.7,
                        metadata={"source": "visual_classifier"}
                    )
                )
            
            if self._detect_sketches(image_array):
                elements.append(
                    ParsedElement(
                        element_type="sketch",
                        content="hand_drawn_elements",
                        confidence=0.6,
                        metadata={"source": "visual_classifier"}
                    )
                )
            
            return elements
            
        except Exception as e:
            self.logger.error(f"Visual classification failed: {e}")
            return []
    
    def _detect_arrows(self, image_array: np.ndarray) -> bool:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        return lines is not None and len(lines) > 0
    
    def _detect_sketches(self, image_array: np.ndarray) -> bool:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 5
    
    def _mock_ocr_results(self) -> List[ParsedElement]:
        return [
            ParsedElement(
                element_type="text",
                content="Mock OCR text content",
                confidence=0.8,
                metadata={"source": "mock_ocr"}
            )
        ]