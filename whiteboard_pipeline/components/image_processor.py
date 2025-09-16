"""
Image Processor Component  
Handles computer vision analysis of images for shape, text, and arrow detection
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import base64
import io
from PIL import Image

from ..models import WhiteboardInput, ParsedInput, ParsedElement


class ImageProcessor:
    """Process images using computer vision techniques"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.min_contour_area = config.get('min_contour_area', 100)
        self.max_contour_area = config.get('max_contour_area', 50000)
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 150)
        
        # Initialize OCR engines
        self.paddleocr_available = False
        self.easyocr_available = False
        
        try:
            import paddleocr
            self.paddleocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.paddleocr_available = True
            self.logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"PaddleOCR not available: {e}")
        
        try:
            import easyocr
            self.easyocr = easyocr.Reader(['en'])
            self.easyocr_available = True
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"EasyOCR not available: {e}")
    
    async def process_image(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Process image using computer vision analysis"""
        # Load image
        image = self._load_image(whiteboard_input)
        if image is None:
            return self._create_empty_result("Failed to load image")
        
        # Convert to OpenCV format
        cv_image = self._pil_to_cv2(image)
        
        # Perform various computer vision analyses
        elements = []
        raw_text_parts = []
        
        # 1. Shape detection
        try:
            shape_elements = await self._detect_shapes(cv_image)
            elements.extend(shape_elements)
            self.logger.info(f"Detected {len(shape_elements)} shapes")
        except Exception as e:
            self.logger.error(f"Shape detection failed: {e}")
        
        # 2. Arrow detection
        try:
            arrow_elements = await self._detect_arrows(cv_image)
            elements.extend(arrow_elements)
            self.logger.info(f"Detected {len(arrow_elements)} arrows")
        except Exception as e:
            self.logger.error(f"Arrow detection failed: {e}")
        
        # 3. Text region detection
        try:
            text_region_elements = await self._detect_text_regions(cv_image)
            elements.extend(text_region_elements)
            self.logger.info(f"Detected {len(text_region_elements)} text regions")
        except Exception as e:
            self.logger.error(f"Text region detection failed: {e}")
        
        # 4. OCR text extraction
        try:
            ocr_elements, ocr_text = await self._extract_text_ocr(image)
            elements.extend(ocr_elements)
            raw_text_parts.append(ocr_text)
            self.logger.info(f"OCR extracted {len(ocr_elements)} text elements")
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
        
        # Combine all text
        combined_text = ' '.join(filter(None, raw_text_parts))
        if not combined_text and elements:
            # Create summary text from detected elements
            shape_count = len([e for e in elements if e.element_type == "shape"])
            text_count = len([e for e in elements if e.element_type == "text"])
            arrow_count = len([e for e in elements if e.element_type == "arrow"])
            combined_text = f"Image analyzed: {shape_count} shapes, {text_count} text regions, {arrow_count} arrows detected"
        
        return ParsedInput(
            elements=elements,
            raw_text=combined_text,
            metadata={
                "input_type": "image",
                "total_elements": len(elements),
                "shapes_detected": len([e for e in elements if e.element_type == "shape"]),
                "text_regions_detected": len([e for e in elements if e.element_type == "text"]),
                "arrows_detected": len([e for e in elements if e.element_type == "arrow"]),
                "processing_method": "computer_vision",
                "image_size": f"{cv_image.shape[1]}x{cv_image.shape[0]}"
            }
        )
    
    def _load_image(self, whiteboard_input: WhiteboardInput) -> Image.Image:
        """Load image from various input formats"""
        try:
            if isinstance(whiteboard_input.content, Path):
                return Image.open(whiteboard_input.content).convert('RGB')
            elif isinstance(whiteboard_input.content, str):
                # Assume base64 encoded
                image_data = base64.b64decode(whiteboard_input.content)
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            elif isinstance(whiteboard_input.content, bytes):
                return Image.open(io.BytesIO(whiteboard_input.content)).convert('RGB')
            else:
                self.logger.error(f"Unsupported image content type: {type(whiteboard_input.content)}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL image to OpenCV format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    async def _detect_shapes(self, image: np.ndarray) -> List[ParsedElement]:
        """Detect shapes in the image using contour analysis"""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, self.canny_low, self.canny_high)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_contour_area <= area <= self.max_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Classify shape type
                    shape_type = self._classify_shape(contour)
                    
                    elements.append(ParsedElement(
                        element_type="shape",
                        content=f"{shape_type} shape at ({x},{y}) size {w}x{h}",
                        confidence=0.7,
                        metadata={
                            "bounding_box": (x, y, w, h),
                            "area": area,
                            "shape_type": shape_type,
                            "source": "computer_vision"
                        }
                    ))
            
            self.logger.info(f"Detected {len(elements)} shapes")
            
        except Exception as e:
            self.logger.error(f"Shape detection failed: {e}")
        
        return elements
    
    def _classify_shape(self, contour) -> str:
        """Classify shape based on contour properties"""
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding box for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Classify based on number of vertices and aspect ratio
        vertices = len(approx)
        
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif vertices > 8:
            # Check if it's roughly circular
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return "circle"
        
        return "polygon"
    
    async def _detect_arrows(self, image: np.ndarray) -> List[ParsedElement]:
        """Detect arrows using line detection and analysis"""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Use HoughLinesP to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Check if this could be an arrow (heuristic)
                    if length > 20:  # Minimum length for arrows
                        # Create bounding box for the line
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = abs(x2 - x1) + 1
                        h = abs(y2 - y1) + 1
                        
                        elements.append(ParsedElement(
                            element_type="arrow",
                            content=f"Line/arrow from ({x1},{y1}) to ({x2},{y2})",
                            confidence=0.6,
                            metadata={
                                "bounding_box": (x, y, w, h),
                                "start_point": (x1, y1),
                                "end_point": (x2, y2),
                                "length": length,
                                "angle": angle,
                                "source": "computer_vision"
                            }
                        ))
            
            self.logger.info(f"Detected {len(elements)} potential arrows")
            
        except Exception as e:
            self.logger.error(f"Arrow detection failed: {e}")
        
        return elements
    
    async def _detect_text_regions(self, image: np.ndarray) -> List[ParsedElement]:
        """Detect text regions using MSER (Maximally Stable Extremal Regions)"""
        elements = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use MSER to detect text regions
            mser = cv2.MSER_create()
            regions, bboxes = mser.detectRegions(gray)
            
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                
                # Filter by size (text regions should be within certain size range)
                if 10 <= w <= 300 and 10 <= h <= 100:
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # Text regions typically have certain aspect ratios
                    if 0.1 <= aspect_ratio <= 10:
                        elements.append(ParsedElement(
                            element_type="text_region",
                            content=f"Text region at ({x},{y}) size {w}x{h}",
                            confidence=0.5,
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
    
    async def _extract_text_ocr(self, image: Image.Image) -> Tuple[List[ParsedElement], str]:
        """Extract text using OCR engines"""
        elements = []
        extracted_texts = []
        
        # Try PaddleOCR first
        if self.paddleocr_available:
            try:
                # Convert PIL to numpy array for PaddleOCR
                img_array = np.array(image)
                result = self.paddleocr.ocr(img_array, cls=True)
                
                if result and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            bbox = line[0]  # Bounding box points
                            text_info = line[1]  # (text, confidence)
                            
                            if len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                                
                                if confidence > 0.3 and text.strip():
                                    extracted_texts.append(text.strip())
                                    
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
        
        # Try EasyOCR as fallback
        if self.easyocr_available and not elements:
            try:
                img_array = np.array(image)
                result = self.easyocr.readtext(img_array)
                
                for detection in result:
                    if len(detection) >= 3:
                        bbox = detection[0]  # Bounding box points
                        text = detection[1]   # Extracted text
                        confidence = detection[2]  # Confidence score
                        
                        if confidence > 0.3 and text.strip():
                            extracted_texts.append(text.strip())
                            
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
        
        combined_text = ' '.join(extracted_texts)
        return elements, combined_text
    
    def _create_empty_result(self, error_message: str) -> ParsedInput:
        """Create empty result with error information"""
        return ParsedInput(
            elements=[
                ParsedElement(
                    element_type="error",
                    content=error_message,
                    confidence=0.0,
                    metadata={"source": "image_processor_error"}
                )
            ],
            raw_text=error_message,
            metadata={
                "input_type": "image",
                "processing_method": "error",
                "error": True
            }
        )