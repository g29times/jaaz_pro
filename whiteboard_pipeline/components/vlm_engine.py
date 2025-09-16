"""
VLM (Vision-Language Model) Engine Component
Handles intent extraction and semantic understanding
"""

import logging
from typing import Dict, Any
import asyncio

from ..models import ParsedInput, SemanticIntent


class VLMEngine:
    """Vision-Language Model engine for intent extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.fallback_enabled = config.get('fallback_enabled', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # VLM provider configuration
        self.provider = config.get('provider', 'openai')
        self.model = config.get('model', 'gpt-4-vision-preview')
        self.api_key = config.get('api_key')
        
        self.logger.info(f"VLM Engine initialized with provider: {self.provider}")
    
    async def extract_intent(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Extract intent (alias for process method for compatibility)"""
        return await self.process(parsed_input)
    
    async def process(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Extract semantic intent from parsed input"""
        try:
            # For now, use rule-based intent extraction
            # This can be enhanced with actual VLM calls later
            return await self._extract_intent_heuristic(parsed_input)
            
        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            return self._create_fallback_intent(parsed_input)
    
    async def _extract_intent_heuristic(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Extract intent using heuristic rules"""
        content = parsed_input.raw_text.lower()
        elements = parsed_input.elements
        
        # Analyze content for flowchart indicators
        flowchart_indicators = [
            'flowchart', 'flow chart', 'diagram', 'process', 'workflow',
            'start', 'end', 'decision', 'step', 'arrow', 'connect'
        ]
        
        # Check for shape elements (strong indicator of flowchart)
        has_shapes = any(elem.element_type == "shape" for elem in elements)
        has_text_content = any(indicator in content for indicator in flowchart_indicators)
        
        if has_shapes or has_text_content:
            intent = "CREATE_FLOWCHART"
            confidence = 0.9 if has_shapes else 0.7
            
            # Determine flow direction
            flow_direction = "TD"  # Default top-down
            if "left" in content and "right" in content:
                flow_direction = "LR"
            elif "horizontal" in content:
                flow_direction = "LR"
            
            context = {
                "flow_direction": flow_direction,
                "complexity": "medium",
                "has_shapes": has_shapes,
                "elements_count": len(elements),
                "detected_features": [
                    elem.element_type for elem in elements 
                    if elem.element_type in ["shape", "arrow", "text"]
                ]
            }
            
            suggested_tasks = [
                "generate_mermaid_flowchart",
                "analyze_flow_structure", 
                "extract_process_steps"
            ]
            
        else:
            # Generic diagram or unknown intent
            intent = "CREATE_DIAGRAM"
            confidence = 0.6
            context = {
                "flow_direction": "TD",
                "complexity": "simple",
                "elements_count": len(elements)
            }
            suggested_tasks = ["generate_basic_diagram"]
        
        return SemanticIntent(
            intent=intent,
            confidence=confidence,
            context=context,
            suggested_tasks=suggested_tasks
        )
    
    def _create_fallback_intent(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Create fallback intent when extraction fails"""
        return SemanticIntent(
            intent="CREATE_DIAGRAM",
            confidence=0.5,
            context={
                "flow_direction": "TD",
                "complexity": "simple",
                "fallback": True,
                "elements_count": len(parsed_input.elements)
            },
            suggested_tasks=["generate_basic_diagram"]
        )