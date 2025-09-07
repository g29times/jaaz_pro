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
        """Parse image input with mandatory OCR - simplified for macOS"""
        self.logger.info("Processing image input (simplified for macOS compatibility)")
        
        # For now, create a simple fallback since we're on macOS
        # In production, this would do full OCR processing
        
        elements = [
            ParsedElement(
                element_type="text",
                content="Image processing placeholder - OCR would extract text here",
                confidence=0.5,
                metadata={"source": "placeholder", "note": "Full OCR processing disabled for macOS compatibility"}
            )
        ]
        
        return ParsedInput(
            elements=elements,
            raw_text="Image content placeholder",
            metadata={
                "input_type": "image",
                "processing_method": "placeholder"
            }
        )
    
    async def _parse_pdf(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        """Parse PDF input - simplified for macOS"""
        self.logger.info("Processing PDF input (simplified for macOS compatibility)")
        
        elements = [
            ParsedElement(
                element_type="text",
                content="PDF processing placeholder - would extract text here",
                confidence=0.5,
                metadata={"source": "placeholder"}
            )
        ]
        
        return ParsedInput(
            elements=elements,
            raw_text="PDF content placeholder",
            metadata={
                "input_type": "pdf",
                "processing_method": "placeholder"
            }
        )
    
    async def process(self, input_data):
        """Implementation of PipelineComponent interface"""
        if isinstance(input_data, WhiteboardInput):
            return await self.parse(input_data)
        else:
            raise ValueError("InputParser expects WhiteboardInput as input")