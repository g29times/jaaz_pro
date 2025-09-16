"""
Enhanced Image Processing for Phase 2
Handles direct image processing with computer vision and OCR
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
import base64

from ..models import WhiteboardInput, ParsedInput, ParsedElement, InputType


class ImageProcessor:
    """Enhanced image processing with computer vision capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Image processing parameters
        self.min_contour_area = self.config.get('min_contour_area', 100)
        self.max_contour_area = self.config.get('max_contour_area', 50000)
        self.edge_threshold_low = self.config.get('edge_threshold_low', 50)
        self.edge_threshold_high = self.config.get('edge_threshold_high', 150)
        
        # OCR engines (will be initialized if available)
        self.ocr_engine = None
        self.backup_ocr_engine = None
        
        self._initialize_ocr_engines()
        self.logger.info("ImageProcessor initialized with computer vision capabilities")
    
    def _initialize_ocr_engines(self):
        """Initialize OCR engines with graceful fallback"""
        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6
            )
            self.logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"PaddleOCR not available: {e}")
        
        try:
            import easyocr
            self.backup_ocr_engine = easyocr.Reader(['en'], gpu=False)
            self.logger.info("EasyOCR initialized as backup")
        except Exception as e:
            self.logger.warning(f"EasyOCR not available: {e}")
    
    async def process_image(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Main image processing pipeline"""
        self.logger.info(f"Processing image input: {whiteboard_input.input_type}")
        
        try:
            # Load and preprocess image
            image = await self._load_image(whiteboard_input)
            if image is None:
                return self._create_fallback_result("Failed to load image")
            
            # Extract visual elements using computer vision
            visual_elements = await self._extract_visual_elements(image)
            
            # Extract text using OCR
            text_elements = await self._extract_text_with_ocr(image)
            
            # Combine all elements
            all_elements = visual_elements + text_elements
            
            # Generate combined text for downstream processing
            raw_text = self._combine_extracted_text(all_elements)
            
            self.logger.info(f"Successfully processed image: {len(all_elements)} elements extracted")
            
            return ParsedInput(
                elements=all_elements,
                raw_text=raw_text,
                metadata={
                    "input_type": "image",
                    "processing_method": "computer_vision_ocr",
                    "visual_elements": len(visual_elements),
                    "text_elements": len(text_elements),
                    "total_elements": len(all_elements)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return self._create_fallback_result(f"Image processing error: {str(e)}")
    
    async def _load_image(self, whiteboard_input: WhiteboardInput) -> Optional[np.ndarray]:
        """Load image from various input types"""
        try:
            if isinstance(whiteboard_input.content, Path):
                # Load from file path
                image = cv2.imread(str(whiteboard_input.content))
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            elif isinstance(whiteboard_input.content, (bytes, bytearray)):
                # Load from bytes
                nparr = np.frombuffer(whiteboard_input.content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            elif isinstance(whiteboard_input.content, str):
                # Try to decode as base64
                try:
                    image_data = base64.b64decode(whiteboard_input.content)
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    # If not base64, might be a file path string
                    image = cv2.imread(whiteboard_input.content)
                    if image is not None:
                        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None
    
    async def _extract_visual_elements(self, image: np.ndarray) -> List[ParsedElement]:
        """Extract visual elements using computer vision"""
        elements = []
        
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect shapes and diagrams
            shapes = await self._detect_shapes(gray)
            elements.extend(shapes)
            
            # Detect arrows and connections
            arrows = await self._detect_arrows(gray)
            elements.extend(arrows)
            
            # Detect text regions (for better OCR)
            text_regions = await self._detect_text_regions(gray)
            elements.extend(text_regions)
            
            self.logger.info(f"Extracted {len(elements)} visual elements")
            
        except Exception as e:
            self.logger.error(f"Visual element extraction failed: {e}")
        
        return elements
    
    async def _detect_shapes(self, gray_image: np.ndarray) -> List[ParsedElement]:
        """Detect geometric shapes in the image"""
        elements = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, self.edge_threshold_low, self.edge_threshold_high)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area < area < self.max_contour_area:
                    # Analyze shape
                    shape_type = self._classify_shape(contour)
                    
                    if shape_type:
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        elements.append(ParsedElement(
                            element_type="shape",
                            content=f"{shape_type} at ({x}, {y}) size {w}x{h}",
                            confidence=0.8,
                            metadata={
                                "shape_type": shape_type,
                                "bounding_box": (x, y, w, h),
                                "area": area,
                                "source": "computer_vision"
                            }
                        ))
            
            self.logger.info(f"Detected {len(elements)} shapes")
            
        except Exception as e:
            self.logger.error(f"Shape detection failed: {e}")
        
        return elements
    
    def _classify_shape(self, contour) -> Optional[str]:
        """Classify detected shape"""
        try:
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            vertices = len(approx)
            
            if vertices == 3:
                return "triangle"
            elif vertices == 4:
                # Check if rectangle or square
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    return "square"
                else:
                    return "rectangle"
            elif vertices > 8:
                # Check if circle
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        return "circle"
            
            return "polygon"
            
        except Exception:
            return None
    
    async def _detect_arrows(self, gray_image: np.ndarray) -> List[ParsedElement]:
        """Detect arrows and connections"""
        elements = []
        
        try:
            # Use line detection to find potential arrows
            edges = cv2.Canny(gray_image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length > 50:  # Only consider longer lines as potential arrows
                        elements.append(ParsedElement(
                            element_type="arrow",
                            content=f"Connection from ({x1}, {y1}) to ({x2}, {y2})",
                            confidence=0.6,
                            metadata={
                                "start_point": (x1, y1),
                                "end_point": (x2, y2),
                                "length": length,
                                "source": "computer_vision"
                            }
                        ))
            
            self.logger.info(f"Detected {len(elements)} potential arrows/connections")
            
        except Exception as e:
            self.logger.error(f"Arrow detection failed: {e}")
        
        return elements
    
    async def _detect_text_regions(self, gray_image: np.ndarray) -> List[ParsedElement]:
        """Detect text regions for better OCR targeting"""
        elements = []
        
        try:
            # Use MSER (Maximally Stable Extremal Regions) for text detection
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray_image)
            
            for region in regions:
                if len(region) > 20:  # Filter small regions
                    # Get bounding rectangle
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    x, y, w, h = cv2.boundingRect(hull)
                    
                    # Check if region looks like text (aspect ratio)
                    aspect_ratio = float(w) / h
                    if 0.2 < aspect_ratio < 10:  # Text-like aspect ratios
                        elements.append(ParsedElement(
                            element_type="text_region",
                            content=f"Text region at ({x}, {y}) size {w}x{h}",
                            confidence=0.7,
                            metadata={
                                "bounding_box": (x, y, w, h),
                                "aspect_ratio": aspect_ratio,
                                "source": "computer_vision"
                            }
                        ))
            
            self.logger.info(f"Detected {len(elements)} text regions")
            
        except Exception as e:
            self.logger.error(f"Text region detection failed: {e}")
        
        return elements
    
    async def _extract_text_with_ocr(self, image: np.ndarray) -> List[ParsedElement]:
        """Extract text using OCR engines"""
        elements = []
        
        # Try primary OCR engine
        if self.ocr_engine:
            try:
                result = self.ocr_engine.ocr(image, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        bbox = line[0]
                        
                        if confidence > 0.3 and text.strip():
                            elements.append(ParsedElement(
                                element_type="text",
                                content=text.strip(),
                                confidence=confidence,
                                metadata={
                                    "bounding_box": bbox,
                                    "ocr_engine": "paddleocr",
                                    "source": "ocr"
                                }
                            ))
                
                self.logger.info(f"PaddleOCR extracted {len(elements)} text elements")
                
            except Exception as e:
                self.logger.warning(f"PaddleOCR failed: {e}")
        
        # Try backup OCR if primary failed or found nothing
        if not elements and self.backup_ocr_engine:
            try:
                result = self.backup_ocr_engine.readtext(image)
                for (bbox, text, confidence) in result:
                    if confidence > 0.3 and text.strip():
                        elements.append(ParsedElement(
                            element_type="text",
                            content=text.strip(),
                            confidence=confidence,
                            metadata={
                                "bounding_box": bbox,
                                "ocr_engine": "easyocr",
                                "source": "ocr"
                            }
                        ))
                
                self.logger.info(f"EasyOCR extracted {len(elements)} text elements")
                
            except Exception as e:
                self.logger.warning(f"EasyOCR failed: {e}")
        
        # If both OCR engines failed, create a placeholder
        if not elements:
            elements.append(ParsedElement(
                element_type="text",
                content="[OCR extraction unavailable - fallback mode]",
                confidence=0.1,
                metadata={
                    "source": "fallback",
                    "note": "OCR engines not available or failed"
                }
            ))
        
        return elements
    
    def _combine_extracted_text(self, elements: List[ParsedElement]) -> str:
        """Combine all extracted text elements into a single string"""
        text_parts = []
        
        # Separate text elements from visual elements
        text_elements = [e for e in elements if e.element_type == "text"]
        visual_elements = [e for e in elements if e.element_type != "text"]
        
        # Add text content
        for element in text_elements:
            text_parts.append(element.content)
        
        # Add visual element descriptions
        if visual_elements:
            text_parts.append("\n--- Visual Elements Detected ---")
            for element in visual_elements:
                text_parts.append(f"{element.element_type}: {element.content}")
        
        return "\n".join(text_parts)
    
    def _create_fallback_result(self, error_message: str) -> ParsedInput:
        """Create fallback result when processing fails"""
        return ParsedInput(
            elements=[
                ParsedElement(
                    element_type="error",
                    content=f"Image processing failed: {error_message}",
                    confidence=0.0,
                    metadata={"source": "fallback", "error": error_message}
                )
            ],
            raw_text=f"Image processing failed: {error_message}",
            metadata={
                "input_type": "image",
                "processing_method": "fallback",
                "error": error_message
            }
        )