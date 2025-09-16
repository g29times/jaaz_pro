"""
Input Parser Component
Handles parsing of various input types including images, PDFs, and text
"""

import logging
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from ..models import WhiteboardInput, ParsedInput, ParsedElement, InputType


class InputParser:
    """Parse various input types into structured format"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize image processor if available
        try:
            from .image_processor import ImageProcessor
            self.image_processor = ImageProcessor(self.config.get('image_processor', {}))
            self.has_image_processor = True
        except ImportError:
            self.logger.warning("Image processor not available")
            self.has_image_processor = False
        
        # OCR configuration
        self.ocr_confidence_threshold = config.get('ocr_confidence_threshold', 0.3)
        
    async def parse(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse input (alias for process method for compatibility)"""
        return await self.process(whiteboard_input)
    
    async def process(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Process input and return structured parsed data"""
        if whiteboard_input.input_type == InputType.TEXT:
            return await self._process_text_input(whiteboard_input)
        elif whiteboard_input.input_type == InputType.IMAGE:
            return await self._process_image_input(whiteboard_input)
        elif whiteboard_input.input_type == InputType.PDF:
            return await self._process_pdf_input(whiteboard_input)
        else:
            raise ValueError(f"Unsupported input type: {whiteboard_input.input_type}")
    
    async def _process_text_input(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Process text input"""
        content = str(whiteboard_input.content)
        
        elements = [
            ParsedElement(
                element_type="text",
                content=content,
                confidence=1.0,
                metadata={"source": "direct_text_input"}
            )
        ]
        
        return ParsedInput(
            elements=elements,
            raw_text=content,
            metadata={
                "input_type": "text",
                "content_length": len(content),
                "processing_method": "direct"
            }
        )
    
    async def _process_image_input(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Process image input with computer vision and OCR"""
        all_elements = []
        raw_text_parts = []
        
        if self.has_image_processor:
            # Use computer vision analysis
            try:
                parsed_input = await self.image_processor.process_image(whiteboard_input)
                all_elements.extend(parsed_input.elements)
                raw_text_parts.append(parsed_input.raw_text)
            except Exception as e:
                self.logger.error(f"Image processing failed: {e}")
        
        # Fallback: basic image processing
        if not all_elements:
            try:
                basic_elements = await self._basic_image_processing(whiteboard_input)
                all_elements.extend(basic_elements)
            except Exception as e:
                self.logger.error(f"Basic image processing failed: {e}")
        
        # Combine all text
        combined_text = ' '.join(filter(None, raw_text_parts))
        if not combined_text:
            combined_text = f"Image processed with {len(all_elements)} elements detected"
        
        return ParsedInput(
            elements=all_elements,
            raw_text=combined_text,
            metadata={
                "input_type": "image",
                "elements_detected": len(all_elements),
                "processing_method": "computer_vision" if self.has_image_processor else "basic"
            }
        )
    
    async def _basic_image_processing(self, whiteboard_input: WhiteboardInput) -> List[ParsedElement]:
        """Basic image processing without computer vision"""
        return [
            ParsedElement(
                element_type="image",
                content="Image content processed",
                confidence=0.5,
                metadata={"source": "basic_image_processing"}
            )
        ]
    
    async def _process_pdf_input(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Process PDF input"""
        all_elements = []
        
        try:
            import fitz  # PyMuPDF
            
            if isinstance(whiteboard_input.content, Path):
                pdf_path = whiteboard_input.content
            else:
                # Handle bytes or base64 content
                raise NotImplementedError("PDF bytes processing not implemented")
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text from page
                page_text = page.get_text()
                if page_text.strip():
                    all_elements.append(ParsedElement(
                        element_type="text",
                        content=page_text.strip(),
                        confidence=0.9,
                        metadata={
                            "source": "pdf_text_extraction",
                            "page": page_num + 1
                        }
                    ))
                
                # Extract images from page
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    all_elements.append(ParsedElement(
                        element_type="image",
                        content=f"PDF image {img_index + 1} on page {page_num + 1}",
                        confidence=0.8,
                        metadata={
                            "source": "pdf_image_extraction",
                            "page": page_num + 1,
                            "image_index": img_index
                        }
                    ))
            
            doc.close()
            
        except ImportError:
            self.logger.warning("PyMuPDF not available, using basic PDF processing")
            all_elements.append(ParsedElement(
                element_type="document",
                content="PDF document processed",
                confidence=0.5,
                metadata={"source": "basic_pdf_processing"}
            ))
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            all_elements.append(ParsedElement(
                element_type="error",
                content=f"PDF processing error: {str(e)}",
                confidence=0.0,
                metadata={"source": "pdf_error"}
            ))
        
        combined_text = ' '.join([elem.content for elem in all_elements if elem.element_type == "text"])
        
        return ParsedInput(
            elements=all_elements,
            raw_text=combined_text,
            metadata={
                "input_type": "pdf",
                "elements_detected": len(all_elements),
                "processing_method": "pymupdf"
            }
        )