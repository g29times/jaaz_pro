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
from .image_processor import ImageProcessor


class InputParser(InputParserInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # OCR is now optional for Phase 2 (prioritize new capabilities)
        self.ocr_engine = None
        self.ocr_confidence_threshold = self.config.get('ocr_confidence_threshold', 0.3)
        self.backup_ocr_engine = None
        
        # Initialize enhanced image processor
        self.image_processor = ImageProcessor(self.config.get('image_processor', {}))
        
        # Enhanced logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info("Initializing InputParser with enhanced image processing capabilities")
        
        self._initialize_ocr_engines()
        
        # Visual element classification parameters
        self.element_types = {
            'text': 0.8,
            'arrow': 0.7, 
            'sketch': 0.6,
            'diagram': 0.7,
            'table': 0.5,
            'flowchart': 0.8,
            'shape': 0.7,  # New for Phase 2
            'text_region': 0.6  # New for Phase 2
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
            self.logger.warning(
                "No OCR engines available - running in fallback mode without OCR. "
                "For production use, install PaddleOCR or EasyOCR:\n"
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
        """Parse image input with enhanced computer vision and OCR"""
        self.logger.info("Processing image input with Phase 2 enhanced capabilities")
        
        try:
            # Use the new ImageProcessor for comprehensive image analysis
            result = await self.image_processor.process_image(whiteboard_input)
            
            # Enhance metadata with processing info
            if result.metadata:
                result.metadata.update({
                    "input_parser_version": "phase_2_enhanced",
                    "capabilities": ["computer_vision", "ocr", "shape_detection", "text_extraction"]
                })
            
            # Log processing results
            if result.elements:
                element_summary = {}
                for element in result.elements:
                    element_type = element.element_type
                    element_summary[element_type] = element_summary.get(element_type, 0) + 1
                
                self.logger.info(f"Image processing completed: {element_summary}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced image processing failed, falling back: {e}")
            
            # Fallback to simple processing
            elements = [
                ParsedElement(
                    element_type="text",
                    content="Image processing fallback - computer vision temporarily unavailable",
                    confidence=0.3,
                    metadata={
                        "source": "fallback", 
                        "error": str(e),
                        "note": "Enhanced image processing failed, using fallback mode"
                    }
                )
            ]
            
            return ParsedInput(
                elements=elements,
                raw_text="Image processing fallback mode",
                metadata={
                    "input_type": "image",
                    "processing_method": "fallback",
                    "error": str(e)
                }
            )
    
    async def _parse_pdf(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse PDF input with enhanced capabilities - extract images and text"""
        self.logger.info("Processing PDF input with Phase 2 enhanced capabilities")
        
        try:
            # Try to extract images from PDF
            import fitz  # PyMuPDF
            
            pdf_path = None
            if isinstance(whiteboard_input.content, Path):
                pdf_path = str(whiteboard_input.content)
            elif isinstance(whiteboard_input.content, str) and Path(whiteboard_input.content).exists():
                pdf_path = whiteboard_input.content
            
            if pdf_path:
                doc = fitz.open(pdf_path)
                all_elements = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text from page
                    page_text = page.get_text()
                    if page_text.strip():
                        all_elements.append(ParsedElement(
                            element_type="text",
                            content=page_text.strip(),
                            confidence=0.9,
                            metadata={
                                "source": "pdf_text_extraction",
                                "page_number": page_num + 1
                            }
                        ))
                    
                    # Extract images from page
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("ppm")
                                
                                # Create WhiteboardInput for the extracted image
                                img_input = WhiteboardInput(
                                    content=img_data,
                                    input_type=InputType.IMAGE,
                                    metadata={
                                        "source": "pdf_extraction",
                                        "page_number": page_num + 1,
                                        "image_index": img_index
                                    }
                                )
                                
                                # Process the extracted image
                                img_result = await self.image_processor.process_image(img_input)
                                
                                # Add extracted elements with PDF context
                                for element in img_result.elements:
                                    element.metadata.update({
                                        "pdf_page": page_num + 1,
                                        "pdf_image_index": img_index
                                    })
                                    all_elements.append(element)
                            
                            pix = None  # Clean up
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to process image {img_index} from page {page_num + 1}: {e}")
                
                doc.close()
                
                # Combine all text
                raw_text = "\n".join([e.content for e in all_elements if e.element_type == "text"])
                
                self.logger.info(f"PDF processing completed: {len(all_elements)} elements from {len(doc)} pages")
                
                return ParsedInput(
                    elements=all_elements,
                    raw_text=raw_text,
                    metadata={
                        "input_type": "pdf",
                        "processing_method": "enhanced_pdf_extraction",
                        "total_pages": len(doc),
                        "total_elements": len(all_elements)
                    }
                )
            
        except ImportError:
            self.logger.warning("PyMuPDF not available, using fallback PDF processing")
        except Exception as e:
            self.logger.error(f"Enhanced PDF processing failed: {e}")
        
        # Fallback processing
        elements = [
            ParsedElement(
                element_type="text",
                content="PDF processing fallback - enhanced PDF extraction temporarily unavailable",
                confidence=0.3,
                metadata={
                    "source": "fallback",
                    "note": "Enhanced PDF processing not available, using fallback mode"
                }
            )
        ]
        
        return ParsedInput(
            elements=elements,
            raw_text="PDF content fallback",
            metadata={
                "input_type": "pdf",
                "processing_method": "fallback"
            }
        )
    
    async def process(self, input_data):
        """Implementation of PipelineComponent interface"""
        if isinstance(input_data, WhiteboardInput):
            return await self.parse(input_data)
        else:
            raise ValueError("InputParser expects WhiteboardInput as input")