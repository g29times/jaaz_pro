import json
import asyncio
from typing import Dict, Any, List, Optional
import base64
import logging

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None

from ..interfaces import VLMEngineInterface
from ..models import ParsedInput, SemanticIntent, ParsedElement


class VLMEngine(VLMEngineInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # vLLM configuration
        self.model_name = self.config.get('model_name', 'Qwen/Qwen-VL-Chat')
        self.tensor_parallel_size = self.config.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = self.config.get('gpu_memory_utilization', 0.9)
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.1)
        
        # Logging setup
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Initializing VLMEngine with model: {self.model_name}")
        
        # Initialize vLLM engine
        self.engine = None
        self.sampling_params = None
        self._initialize_vllm_engine()
    
    def _initialize_vllm_engine(self):
        """Initialize vLLM engine for production inference"""
        if not VLLM_AVAILABLE:
            self.logger.warning("vLLM not available, will use fallback mode")
            return
        
        try:
            # Configure vLLM engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=False,  # Use CUDA graphs for better performance
                disable_log_stats=False,  # Keep stats for monitoring
                max_model_len=None,  # Use model's default
                trust_remote_code=True  # Required for Qwen-VL
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.8,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stop_token_ids=None
            )
            
            self.logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vLLM engine: {e}")
            self.engine = None
            
    async def extract_intent(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Extract semantic intent using vLLM engine"""
        self.logger.info("Extracting semantic intent from parsed input")
        
        # Log input details for feedback
        self.logger.info(f"Input elements: {len(parsed_input.elements)}")
        self.logger.info(f"Raw text length: {len(parsed_input.raw_text)}")
        for i, element in enumerate(parsed_input.elements):
            self.logger.debug(f"Element {i}: type={element.element_type}, confidence={element.confidence}")
        
        prompt = self._build_intent_prompt(parsed_input)
        
        if self.engine is not None:
            try:
                intent_data = await self._call_vllm_engine(prompt, parsed_input)
                
                result = SemanticIntent(
                    intent=intent_data.get('intent', 'unknown'),
                    confidence=intent_data.get('confidence', 0.5),
                    context=intent_data.get('context', {}),
                    suggested_tasks=intent_data.get('suggested_tasks', [])
                )
                
                # Log result for feedback
                self.logger.info(f"Intent extracted: {result.intent} (confidence: {result.confidence})")
                self.logger.debug(f"Suggested tasks: {result.suggested_tasks}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"vLLM inference failed: {e}, falling back to rule-based extraction")
                return self._fallback_intent_extraction(parsed_input)
        else:
            self.logger.info("vLLM engine not available, using fallback intent extraction")
            return self._fallback_intent_extraction(parsed_input)
    
    async def _call_vllm_engine(self, prompt: str, parsed_input: ParsedInput) -> Dict[str, Any]:
        """Call vLLM engine for inference"""
        
        # Prepare the request
        request_id = f"intent_extraction_{id(parsed_input)}"
        
        # For vision-language models, we might need to include image data
        inputs = self._prepare_multimodal_inputs(prompt, parsed_input)
        
        self.logger.debug(f"Sending inference request: {request_id}")
        
        # Generate response
        results = []
        async for result in self.engine.generate(
            inputs,
            self.sampling_params,
            request_id=request_id
        ):
            results.append(result)
        
        if not results:
            raise Exception("No results returned from vLLM engine")
        
        # Get the final result
        final_result = results[-1]
        
        if not final_result.outputs:
            raise Exception("No outputs in vLLM result")
        
        output_text = final_result.outputs[0].text
        
        self.logger.debug(f"vLLM response length: {len(output_text)}")
        
        # Parse the response
        return self._parse_vlm_response({'content': output_text})
    
    def _prepare_multimodal_inputs(self, prompt: str, parsed_input: ParsedInput) -> str:
        """Prepare inputs for multimodal VLM"""
        
        # For now, focus on text-based processing
        # In future versions, this can be extended to handle images directly
        
        return prompt
    
    def _build_intent_prompt(self, parsed_input: ParsedInput) -> str:
        """Build optimized prompt for intent extraction"""
        
        elements_summary = self._summarize_elements(parsed_input.elements)
        
        prompt = f"""You are a whiteboard content analyzer specializing in converting sketches and diagrams to Mermaid flowcharts.

CONTENT ANALYSIS:
Raw text extracted: {parsed_input.raw_text[:500]}
Visual elements detected: {elements_summary}
Total elements: {len(parsed_input.elements)}

TASK: Determine the user's primary intent for this whiteboard content.

Focus on these key intents:
1. CREATE_FLOWCHART - User wants to convert sketches/text into a Mermaid flowchart
2. ANALYZE_PROCESS - User wants to understand workflow or process steps
3. DOCUMENT_WORKFLOW - User wants to create documentation from their sketch

Respond with JSON in this exact format:
{{
    "intent": "CREATE_FLOWCHART|ANALYZE_PROCESS|DOCUMENT_WORKFLOW",
    "confidence": 0.0-1.0,
    "context": {{
        "domain": "business|technical|creative|process",
        "complexity": "low|medium|high",
        "flow_direction": "top-down|left-right|mixed",
        "key_concepts": ["concept1", "concept2"],
        "has_decision_points": true|false,
        "has_process_steps": true|false
    }},
    "suggested_tasks": ["create_mermaid_flowchart"]
}}

IMPORTANT: Always prioritize CREATE_FLOWCHART intent when process steps, arrows, or workflow elements are detected."""
        
        return prompt
    
    def _summarize_elements(self, elements: List[ParsedElement]) -> str:
        """Summarize detected elements for logging and analysis"""
        element_counts = {}
        high_confidence_elements = []
        
        for element in elements:
            element_counts[element.element_type] = element_counts.get(element.element_type, 0) + 1
            if element.confidence > 0.7:
                high_confidence_elements.append(f"{element.element_type}({element.confidence:.2f})")
        
        summary_parts = []
        for elem_type, count in element_counts.items():
            summary_parts.append(f"{count} {elem_type}")
        
        summary = ", ".join(summary_parts)
        if high_confidence_elements:
            summary += f" | High confidence: {', '.join(high_confidence_elements)}"
        
        return summary
    
    def _parse_vlm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VLM response with improved error handling"""
        try:
            content = response.get('content', '')
            
            # Find JSON in response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                parsed = json.loads(json_content)
                
                # Validate required fields
                if 'intent' not in parsed:
                    raise ValueError("Missing 'intent' field in response")
                
                # Log parsed response for feedback
                self.logger.debug(f"Parsed VLM response: {parsed}")
                
                return parsed
            else:
                self.logger.warning("No valid JSON found in VLM response, using text analysis")
                return self._extract_intent_from_text(content)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to parse VLM response: {e}")
            self.logger.debug(f"Raw response: {response}")
            return self._default_intent_response()
    
    def _extract_intent_from_text(self, text: str) -> Dict[str, Any]:
        """Extract intent from text response when JSON parsing fails"""
        text_lower = text.lower()
        
        # Enhanced intent detection focusing on flowchart creation
        if any(word in text_lower for word in ['flowchart', 'mermaid', 'diagram', 'flow', 'process', 'workflow']):
            intent = "CREATE_FLOWCHART"
            confidence = 0.7
            tasks = ["create_mermaid_flowchart"]
            context = {
                "domain": "process",
                "complexity": "medium",
                "flow_direction": "top-down",
                "has_process_steps": True
            }
        elif any(word in text_lower for word in ['analyze', 'understand', 'explain']):
            intent = "ANALYZE_PROCESS"
            confidence = 0.6
            tasks = ["analyze_workflow"]
            context = {
                "domain": "business",
                "complexity": "medium"
            }
        elif any(word in text_lower for word in ['document', 'record', 'capture']):
            intent = "DOCUMENT_WORKFLOW"
            confidence = 0.6
            tasks = ["create_documentation"]
            context = {
                "domain": "business",
                "complexity": "low"
            }
        else:
            intent = "CREATE_FLOWCHART"  # Default to flowchart creation
            confidence = 0.4
            tasks = ["create_mermaid_flowchart"]
            context = {
                "domain": "general",
                "complexity": "medium",
                "fallback": True
            }
        
        return {
            "intent": intent,
            "confidence": confidence,
            "context": context,
            "suggested_tasks": tasks
        }
    
    def _fallback_intent_extraction(self, parsed_input: ParsedInput) -> SemanticIntent:
        """Enhanced fallback intent extraction focused on flowchart creation"""
        self.logger.info("Using fallback intent extraction")
        
        text_content = parsed_input.raw_text.lower()
        element_types = [elem.element_type for elem in parsed_input.elements]
        
        # Log fallback analysis for feedback
        self.logger.info(f"Fallback analysis - text length: {len(text_content)}, elements: {len(element_types)}")
        
        # Prioritize flowchart creation based on content analysis
        flowchart_indicators = ['arrow', 'sketch', 'flow', 'process', 'step', 'workflow', 'diagram']
        flowchart_score = sum(1 for indicator in flowchart_indicators 
                             if indicator in text_content or indicator in element_types)
        
        if flowchart_score > 0 or 'arrow' in element_types or 'sketch' in element_types:
            intent = "CREATE_FLOWCHART"
            confidence = min(0.6 + (flowchart_score * 0.1), 0.9)
            tasks = ["create_mermaid_flowchart"]
            context = {
                "domain": "process",
                "complexity": "medium" if flowchart_score > 2 else "low",
                "flow_direction": "top-down",
                "has_process_steps": flowchart_score > 1,
                "has_decision_points": any(word in text_content for word in ['if', 'decide', 'choice', 'yes', 'no']),
                "fallback": True
            }
        else:
            # Still default to flowchart creation
            intent = "CREATE_FLOWCHART"
            confidence = 0.3
            tasks = ["create_mermaid_flowchart"]
            context = {
                "domain": "general",
                "complexity": "low",
                "flow_direction": "top-down",
                "fallback": True
            }
        
        # Log fallback decision for feedback
        self.logger.info(f"Fallback decision: {intent} (confidence: {confidence})")
        
        return SemanticIntent(
            intent=intent,
            confidence=confidence,
            context=context,
            suggested_tasks=tasks
        )
    
    def _default_intent_response(self) -> Dict[str, Any]:
        """Default response when all parsing fails"""
        return {
            "intent": "CREATE_FLOWCHART",
            "confidence": 0.2,
            "context": {
                "domain": "unknown",
                "complexity": "medium",
                "fallback": True,
                "parsing_failed": True
            },
            "suggested_tasks": ["create_mermaid_flowchart"]
        }
    
    async def cleanup(self):
        """Cleanup vLLM engine resources"""
        if self.engine is not None:
            try:
                # vLLM doesn't have explicit cleanup, but we can log
                self.logger.info("Cleaning up vLLM engine resources")
                self.engine = None
            except Exception as e:
                self.logger.error(f"Error during vLLM cleanup: {e}")