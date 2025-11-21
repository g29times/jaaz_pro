"""
Mermaid Flow Generator Component
Generates Mermaid diagrams from parsed input with intelligent analysis capabilities
"""

import logging
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..models import TaskStep, GeneratorOutput, ParsedElement


class MermaidFlowGenerator:
    """Generate Mermaid flowchart diagrams"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.fallback_enabled = config.get('fallback_enabled', True)
        self.use_intelligent_analysis = config.get('use_intelligent_analysis', True)

        # LLM configuration
        self.llm_provider = config.get('llm_provider', 'gemini')  # Default to Gemini
        self.api_key = config.get('api_key')

        # Initialize Gemini client as PRIMARY LLM
        self.gemini_client = None
        if self.llm_provider == 'gemini':
            try:
                from .gemini_client import GeminiClient
                self.gemini_client = GeminiClient(config)
                self.logger.info("Gemini client initialized as primary LLM provider")
            except ImportError as e:
                self.logger.warning(f"Gemini client not available: {e}")
            except Exception as e:
                self.logger.warning(f"Gemini client initialization failed: {e}")

        # Initialize Ollama client as FALLBACK LLM
        self.ollama_client = None
        try:
            from .ollama_client import OllamaClient
            self.ollama_client = OllamaClient(config)
            self.logger.info("Ollama client initialized as fallback LLM provider")
        except ImportError as e:
            self.logger.warning(f"Ollama client not available: {e}")
        except Exception as e:
            self.logger.warning(f"Ollama client initialization failed: {e}")

        # Initialize intelligent generator
        self.intelligent_generator = None
        if self.use_intelligent_analysis:
            try:
                from .intelligent_mermaid_generator import IntelligentMermaidGenerator
                self.intelligent_generator = IntelligentMermaidGenerator(config)
                self.logger.info("Intelligent Mermaid generator initialized")
            except ImportError as e:
                self.logger.warning(f"Intelligent generator not available: {e}")
                self.use_intelligent_analysis = False

        # Generation statistics
        self.generation_stats = {
            'total_generations': 0,
            'intelligent_generations': 0,
            'llm_generations': 0,
            'fallback_generations': 0
        }

        # Performance tracking
        self.avg_generation_time = 0.0
        self._generation_times = []
    
    async def generate(self, task_step: TaskStep, context: Dict[str, Any]) -> GeneratorOutput:
        """Generate Mermaid diagram from task step and context"""
        session_id = context.get('session_id', 'unknown')
        generation_start = datetime.now()

        try:
            content = task_step.parameters.get('content', '')
            direction = task_step.parameters.get('direction', 'TD')

            self.generation_stats['total_generations'] += 1

            visual_elements = context.get('visual_elements', [])
            input_type = context.get('input_type', 'text')

            # NEW PRIORITY ORDER: Choose strategy based on input type

            # Strategy 1: For TEXT input, use LLM as PRIMARY method
            if input_type == 'text' or not visual_elements:
                self.logger.info(f"[{session_id}] Text input detected, using LLM as primary method")

                # Try LLM first for text understanding (Gemini → Ollama fallback)
                if self.gemini_client or self.ollama_client or (self.llm_provider and self.api_key):
                    self.logger.info(f"[{session_id}] Generating Mermaid with LLM (primary method)")

                    mermaid_code = await self._generate_mermaid_with_llm(content, direction, context, session_id)

                    if mermaid_code:
                        generation_time = (datetime.now() - generation_start).total_seconds()
                        self.generation_stats['llm_generations'] += 1
                        self._update_avg_generation_time(generation_time)

                        output_path = self._create_output_file(mermaid_code, session_id)

                        self.logger.info(f"[{session_id}] LLM generation successful in {generation_time:.2f}s")

                        # Determine which provider was used
                        provider = 'gemini' if self.gemini_client else ('ollama' if self.ollama_client else 'openai')

                        return GeneratorOutput(
                            content=mermaid_code,
                            output_type='mermaid',
                            file_path=output_path,
                            metadata={
                                'generation_method': 'llm',
                                'generator': provider,
                                'model': self.config.get('gemini_model' if provider == 'gemini' else 'ollama_model', 'unknown'),
                                'generation_time': generation_time,
                                'content_length': len(mermaid_code),
                                'output_length': len(mermaid_code.split('\n')),
                                'session_id': session_id,
                                'created_at': datetime.now().isoformat(),
                                'success': True,
                                'priority': 'primary'
                            }
                        )
                    else:
                        self.logger.warning(f"[{session_id}] LLM generation returned empty, falling back")

                # Fallback to rule-based for text input
                self.logger.info(f"[{session_id}] Using rule-based fallback for text input")
                mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)

            # Strategy 2: For IMAGE input with visual shapes, use CV analysis first
            elif self.use_intelligent_analysis and visual_elements:
                self.logger.info(f"[{session_id}] Image input with {len(visual_elements)} visual elements, using CV analysis")

                try:
                    result = await self.intelligent_generator.generate_from_visual_analysis(visual_elements, session_id)

                    if result and result.content:
                        generation_time = (datetime.now() - generation_start).total_seconds()
                        self.generation_stats['intelligent_generations'] += 1
                        self._update_avg_generation_time(generation_time)

                        # Enhanced metadata for intelligent generation
                        result.metadata.update({
                            "generation_time": generation_time,
                            "content_length": len(result.content),
                            "output_length": len(result.content.split('\n')),
                            "session_id": session_id,
                            "created_at": datetime.now().isoformat(),
                            "success": True,
                            "priority": "cv_analysis"
                        })

                        self.logger.info(f"[{session_id}] CV analysis completed in {generation_time:.2f}s")
                        return result

                except Exception as e:
                    self.logger.warning(f"[{session_id}] CV analysis failed: {e}, falling back to LLM")

                # Fallback to LLM for images if CV fails
                if self.gemini_client or self.ollama_client or (self.llm_provider and self.api_key):
                    self.logger.info(f"[{session_id}] Trying LLM fallback for image input")

                    mermaid_code = await self._generate_mermaid_with_llm(content, direction, context, session_id)

                    if mermaid_code:
                        self.generation_stats['llm_generations'] += 1
                        self.logger.info(f"[{session_id}] LLM fallback successful")
                    else:
                        self.logger.warning(f"[{session_id}] LLM fallback failed, using rule-based")
                        mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)
                else:
                    mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)

            else:
                # No LLM, no CV - direct fallback
                self.logger.info(f"[{session_id}] No LLM or CV available, using rule-based generation")
                mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)

            # Create output file
            generation_time = (datetime.now() - generation_start).total_seconds()
            output_path = self._create_output_file(mermaid_code, session_id)

            self._update_avg_generation_time(generation_time)

            metadata = {
                'generation_method': 'llm' if self.ollama_client or self.api_key else 'fallback',
                'generator': 'mermaid_flow_generator',
                'generation_time': generation_time,
                'content_length': len(mermaid_code),
                'output_length': len(mermaid_code.split('\n')),
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'success': True
            }

            return GeneratorOutput(
                content=mermaid_code,
                output_type='mermaid',
                file_path=output_path,
                metadata=metadata
            )

        except Exception as e:
            generation_time = (datetime.now() - generation_start).total_seconds()
            error_msg = f"Mermaid generation failed: {str(e)}"

            self.logger.error(f"[{session_id}] {error_msg}")
            self.logger.error(f"[{session_id}] Generation failed after {generation_time:.2f}s")

            # Log error details for feedback
            error_metadata = {
                'error': True,
                'error_message': str(e),
                'generation_time': generation_time,
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'success': False
            }

            return GeneratorOutput(
                content=f"Error: {error_msg}",
                output_type='error',
                file_path=None,
                metadata=error_metadata
            )
    
    async def _generate_mermaid_with_llm(self, content: str, direction: str, context: Dict[str, Any], session_id: str) -> Optional[str]:
        """Generate Mermaid code using LLM (Gemini primary → Ollama fallback)"""

        visual_elements = context.get('visual_elements', [])
        input_type = context.get('input_type', 'text')

        # Determine if we have actual visual shapes (not just text elements)
        has_visual_shapes = False
        if visual_elements:
            for elem in visual_elements:
                if hasattr(elem, 'element_type'):
                    if elem.element_type in ['shape', 'arrow', 'connector', 'diagram']:
                        has_visual_shapes = True
                        break

        # Try Gemini first (PRIMARY provider)
        if self.gemini_client:
            self.logger.info(f"[{session_id}] Generating Mermaid with Gemini (primary)")

            try:
                if has_visual_shapes and input_type != 'text':
                    # Use elements-based generation for images with shapes
                    self.logger.info(f"[{session_id}] Using Gemini element-based generation")
                    mermaid_code = await self.gemini_client.generate_mermaid_from_elements(
                        visual_elements,
                        {'flow_direction': direction}
                    )
                else:
                    # Use text-based generation for text input
                    self.logger.info(f"[{session_id}] Using Gemini text-based generation")
                    mermaid_code = await self.gemini_client.generate_mermaid_from_text(
                        content,
                        direction
                    )

                if mermaid_code:
                    self.logger.info(f"[{session_id}] Gemini generation successful")
                    return mermaid_code
                else:
                    self.logger.warning(f"[{session_id}] Gemini returned empty result, falling back to Ollama")

            except Exception as e:
                self.logger.warning(f"[{session_id}] Gemini generation failed: {e}, falling back to Ollama")

        # Fallback to Ollama if Gemini fails or unavailable
        if self.ollama_client:
            self.logger.info(f"[{session_id}] Generating Mermaid with Ollama (fallback)")

            try:
                if has_visual_shapes and input_type != 'text':
                    # Use elements-based generation for images with shapes
                    self.logger.info(f"[{session_id}] Using Ollama element-based generation")
                    mermaid_code = await self.ollama_client.generate_mermaid_from_elements(
                        visual_elements,
                        {'flow_direction': direction}
                    )
                else:
                    # Use text-based generation for text input
                    self.logger.info(f"[{session_id}] Using Ollama text-based generation")
                    mermaid_code = await self.ollama_client.generate_mermaid_from_text(
                        content,
                        direction
                    )

                if mermaid_code:
                    self.logger.info(f"[{session_id}] Ollama generation successful")
                    return mermaid_code
                else:
                    self.logger.warning(f"[{session_id}] Ollama returned empty result")

            except Exception as e:
                self.logger.warning(f"[{session_id}] Ollama generation failed: {e}")

        return None
    
    def _generate_fallback_mermaid(self, content: str, direction: str, context: Dict[str, Any], session_id: str) -> str:
        """Generate basic fallback Mermaid diagram"""
        self.generation_stats['fallback_generations'] += 1
        
        # Simple fallback generation
        mermaid_code = f"""flowchart {direction}
    A([Start])
    B[Process Input]
    A --> B
    C([End])
    B --> C"""
        
        self.logger.info(f"[{session_id}] Generated fallback Mermaid diagram")
        return mermaid_code
    
    def _create_output_file(self, mermaid_code: str, session_id: str) -> Path:
        """Create output file for Mermaid code"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mermaid_flowchart_{session_id}_{timestamp}.mmd"
        
        output_path = Path(tempfile.gettempdir()) / filename
        
        with open(output_path, 'w') as f:
            f.write(mermaid_code)
        
        return output_path
    
    def _update_avg_generation_time(self, generation_time: float):
        """Update average generation time"""
        self._generation_times.append(generation_time)
        # Keep only last 100 measurements
        if len(self._generation_times) > 100:
            self._generation_times = self._generation_times[-100:]
        
        self.avg_generation_time = sum(self._generation_times) / len(self._generation_times)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            **self.generation_stats,
            'avg_generation_time': self.avg_generation_time,
            'intelligent_analysis_enabled': self.use_intelligent_analysis,
            'fallback_enabled': self.fallback_enabled
        }