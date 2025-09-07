import asyncio
import aiohttp
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

from ..interfaces import GeneratorInterface
from ..models import TaskStep, GeneratorOutput, GeneratorType


class MermaidFlowGenerator(GeneratorInterface):
    """Enhanced Mermaid generator with comprehensive logging"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm_provider = self.config.get('llm_provider', 'openai')
        self.api_key = self.config.get('api_key')
        self.model_name = self.config.get('model_name', 'gpt-4')
        
        # Enhanced logging setup
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info(f"Initializing MermaidFlowGenerator with {self.llm_provider}")
        
        # Generation statistics for feedback
        self.generation_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'llm_generations': 0,
            'fallback_generations': 0,
            'avg_generation_time': 0.0
        }
        
    def supports_task_type(self, task_type: str) -> bool:
        return task_type in ['create_mermaid_flowchart', 'CREATE_FLOWCHART']
    
    async def generate(self, task_step: TaskStep, context: Dict[str, Any]) -> GeneratorOutput:
        """Generate Mermaid flowchart with comprehensive logging"""
        
        generation_start = datetime.now()
        session_id = context.get('session_id', 'unknown')
        
        self.logger.info(f"[{session_id}] Starting Mermaid generation for task: {task_step.action}")
        
        # Log input parameters for feedback
        parameters = task_step.parameters
        content = parameters.get('content', 'Process flow')
        direction = parameters.get('direction', 'TD')
        theme = parameters.get('theme', 'default')
        
        self.logger.info(f"[{session_id}] Parameters: content_length={len(content)}, direction={direction}, theme={theme}")
        
        # Update stats
        self.generation_stats['total_generations'] += 1
        
        try:
            # Attempt LLM-based generation first
            if self.llm_provider and self.api_key:
                self.logger.debug(f"[{session_id}] Attempting LLM-based Mermaid generation")
                
                mermaid_code = await self._generate_mermaid_with_llm(content, direction, context, session_id)
                
                if mermaid_code:
                    self.generation_stats['llm_generations'] += 1
                    self.logger.info(f"[{session_id}] LLM generation successful")
                else:
                    self.logger.warning(f"[{session_id}] LLM generation returned empty result, using fallback")
                    mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)
                    self.generation_stats['fallback_generations'] += 1
            else:
                self.logger.info(f"[{session_id}] No LLM configured, using structured fallback generation")
                mermaid_code = self._generate_fallback_mermaid(content, direction, context, session_id)
                self.generation_stats['fallback_generations'] += 1
            
            # Save to file
            output_path = await self._save_mermaid_file(mermaid_code, session_id)
            
            generation_time = (datetime.now() - generation_start).total_seconds()
            
            # Update statistics
            self.generation_stats['successful_generations'] += 1
            self._update_avg_generation_time(generation_time)
            
            # Log success metrics for feedback
            self.logger.info(f"[{session_id}] Mermaid generation completed in {generation_time:.2f}s")
            self.logger.info(f"[{session_id}] Generated {len(mermaid_code)} characters of Mermaid code")
            self.logger.info(f"[{session_id}] File saved to: {output_path}")
            
            # Create detailed metadata for feedback
            metadata = {
                'direction': direction,
                'theme': theme,
                'generator': 'mermaid_flow',
                'generation_method': 'llm' if self.generation_stats['llm_generations'] > self.generation_stats['fallback_generations'] else 'fallback',
                'generation_time': generation_time,
                'content_length': len(content),
                'output_length': len(mermaid_code),
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
                metadata=error_metadata
            )
    
    async def _generate_mermaid_with_llm(self, content: str, direction: str, context: Dict[str, Any], session_id: str) -> Optional[str]:
        """Generate Mermaid code using LLM with enhanced prompt"""
        
        # Build enhanced prompt focusing on flowchart creation
        intent_context = context.get('semantic_intent', {})
        parsed_input = context.get('parsed_input', {})
        
        prompt = f"""You are an expert at creating Mermaid flowcharts. Convert the following content into a clear, well-structured Mermaid flowchart.

CONTENT TO CONVERT:
{content[:1000]}

CONTEXT:
- Intent: {getattr(intent_context, 'intent', 'CREATE_FLOWCHART')}
- Confidence: {getattr(intent_context, 'confidence', 'unknown')}
- Flow Direction: {direction}
- Domain: {getattr(intent_context, 'context', {}).get('domain', 'general')}

REQUIREMENTS:
1. Use flowchart {direction} syntax
2. Create clear, descriptive node labels
3. Use appropriate node shapes:
   - Rectangles for processes: A[Process Name]
   - Diamonds for decisions: B{{Decision Point}}
   - Rounded rectangles for start/end: C([Start/End])
4. Connect nodes with arrows and labels
5. Keep it clean and readable
6. Include all key steps from the content

EXAMPLE FORMAT:
flowchart {direction}
    A([Start]) --> B[First Process]
    B --> C{{Decision?}}
    C -->|Yes| D[Action 1]
    C -->|No| E[Action 2]
    D --> F([End])
    E --> F

Return ONLY the Mermaid code, starting with 'flowchart {direction}'."""
        
        try:
            if self.llm_provider == 'openai':
                response = await self._call_openai(prompt, session_id)
            else:
                self.logger.warning(f"[{session_id}] Unsupported LLM provider: {self.llm_provider}")
                return None
            
            # Extract and validate Mermaid code
            mermaid_code = self._extract_mermaid_code(response, session_id)
            
            if mermaid_code and self._validate_mermaid_syntax(mermaid_code, session_id):
                return mermaid_code
            else:
                self.logger.warning(f"[{session_id}] LLM generated invalid Mermaid code")
                return None
                
        except Exception as e:
            self.logger.error(f"[{session_id}] LLM generation error: {e}")
            return None
    
    async def _call_openai(self, prompt: str, session_id: str) -> str:
        """Call OpenAI API with error handling and logging"""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a Mermaid flowchart expert. Generate clean, valid Mermaid syntax."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self.logger.debug(f"[{session_id}] Calling OpenAI API")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    self.logger.debug(f"[{session_id}] OpenAI API success, response length: {len(content)}")
                    return content
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
    
    def _extract_mermaid_code(self, response: str, session_id: str) -> Optional[str]:
        """Extract Mermaid code from LLM response"""
        
        # Look for code blocks
        if "```mermaid" in response:
            start = response.find("```mermaid") + 10
            end = response.find("```", start)
            if end > start:
                code = response[start:end].strip()
                self.logger.debug(f"[{session_id}] Extracted Mermaid from code block")
                return code
        
        # Look for flowchart directive
        if "flowchart" in response:
            lines = response.split('\n')
            mermaid_lines = []
            in_flowchart = False
            
            for line in lines:
                if line.strip().startswith('flowchart'):
                    in_flowchart = True
                    mermaid_lines.append(line.strip())
                elif in_flowchart:
                    if line.strip() and not line.strip().startswith('#'):
                        mermaid_lines.append(line.strip())
                    elif not line.strip() and mermaid_lines:
                        break
            
            if mermaid_lines:
                code = '\n'.join(mermaid_lines)
                self.logger.debug(f"[{session_id}] Extracted Mermaid from flowchart directive")
                return code
        
        self.logger.warning(f"[{session_id}] Could not extract valid Mermaid code from response")
        return None
    
    def _validate_mermaid_syntax(self, mermaid_code: str, session_id: str) -> bool:
        """Basic validation of Mermaid syntax"""
        
        if not mermaid_code.strip():
            return False
        
        # Must start with flowchart directive
        if not mermaid_code.strip().startswith('flowchart'):
            self.logger.warning(f"[{session_id}] Mermaid code doesn't start with 'flowchart'")
            return False
        
        # Should have at least one arrow
        if '-->' not in mermaid_code:
            self.logger.warning(f"[{session_id}] Mermaid code has no arrows (-->)")
            return False
        
        # Basic structure check
        lines = [line.strip() for line in mermaid_code.split('\n') if line.strip()]
        if len(lines) < 2:
            self.logger.warning(f"[{session_id}] Mermaid code too short")
            return False
        
        self.logger.debug(f"[{session_id}] Mermaid syntax validation passed")
        return True
    
    def _generate_fallback_mermaid(self, content: str, direction: str, context: Dict[str, Any], session_id: str) -> str:
        """Generate structured fallback Mermaid when LLM is unavailable"""
        
        self.logger.info(f"[{session_id}] Generating structured fallback Mermaid")
        
        # Analyze content for process steps
        steps = self._extract_process_steps(content, session_id)
        
        # Build Mermaid flowchart
        mermaid_lines = [f"flowchart {direction}"]
        
        if not steps:
            # Generic fallback
            mermaid_lines.extend([
                "    A([Start]) --> B[Process Input]",
                "    B --> C{Decision Point}",
                "    C -->|Yes| D[Execute Action]",
                "    C -->|No| E[Alternative Path]",
                "    D --> F([End])",
                "    E --> F"
            ])
            self.logger.info(f"[{session_id}] Using generic fallback structure")
        else:
            # Build from extracted steps
            node_id = ord('A')
            
            # Start node
            mermaid_lines.append(f"    {chr(node_id)}([Start])")
            prev_node = chr(node_id)
            node_id += 1
            
            # Process steps
            for i, step in enumerate(steps):
                current_node = chr(node_id)
                
                # Determine node type
                if any(word in step.lower() for word in ['decision', 'choose', 'if', 'whether']):
                    node_shape = f"{{{step}}}"
                else:
                    node_shape = f"[{step}]"
                
                mermaid_lines.append(f"    {current_node}{node_shape}")
                mermaid_lines.append(f"    {prev_node} --> {current_node}")
                
                prev_node = current_node
                node_id += 1
            
            # End node
            end_node = chr(node_id)
            mermaid_lines.append(f"    {end_node}([End])")
            mermaid_lines.append(f"    {prev_node} --> {end_node}")
            
            self.logger.info(f"[{session_id}] Built fallback from {len(steps)} extracted steps")
        
        mermaid_code = '\n'.join(mermaid_lines)
        self.logger.debug(f"[{session_id}] Fallback Mermaid generated: {len(mermaid_code)} characters")
        
        return mermaid_code
    
    def _extract_process_steps(self, content: str, session_id: str) -> List[str]:
        """Extract process steps from content for fallback generation"""
        
        steps = []
        
        # Look for numbered lists
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Numbered steps: "1. Step", "2. Step", etc.
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                step = line.lstrip('0123456789.- *').strip()
                if step and len(step) > 3:
                    steps.append(step[:50])  # Limit length
        
        # Look for action words if no numbered steps
        if not steps:
            action_words = ['process', 'analyze', 'create', 'generate', 'validate', 'submit', 'review']
            sentences = content.replace('.', '.\n').split('\n')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in action_words) and len(sentence) > 10:
                    steps.append(sentence[:50])
                    if len(steps) >= 5:  # Limit to reasonable number
                        break
        
        self.logger.debug(f"[{session_id}] Extracted {len(steps)} process steps from content")
        return steps[:8]  # Maximum 8 steps for readability
    
    async def _save_mermaid_file(self, mermaid_code: str, session_id: str) -> Path:
        """Save Mermaid code to file with session tracking"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flowchart_{session_id}_{timestamp}.mmd"
        output_path = Path(tempfile.gettempdir()) / filename
        
        output_path.write_text(mermaid_code, encoding='utf-8')
        
        self.logger.debug(f"[{session_id}] Mermaid file saved: {output_path}")
        return output_path
    
    def _update_avg_generation_time(self, generation_time: float):
        """Update average generation time for statistics"""
        current_avg = self.generation_stats['avg_generation_time']
        total_successful = self.generation_stats['successful_generations']
        
        # Rolling average
        if total_successful == 1:
            self.generation_stats['avg_generation_time'] = generation_time
        else:
            self.generation_stats['avg_generation_time'] = (
                (current_avg * (total_successful - 1) + generation_time) / total_successful
            )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics for feedback analysis"""
        
        stats = self.generation_stats.copy()
        
        if stats['total_generations'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_generations']
            stats['llm_usage_rate'] = stats['llm_generations'] / stats['total_generations']
            stats['fallback_usage_rate'] = stats['fallback_generations'] / stats['total_generations']
        else:
            stats['success_rate'] = 0.0
            stats['llm_usage_rate'] = 0.0
            stats['fallback_usage_rate'] = 0.0
        
        self.logger.info(f"Generation stats: {stats['total_generations']} total, {stats['success_rate']:.1%} success rate")
        
        return stats
    
    async def process(self, input_data):
        """Implementation of PipelineComponent interface"""
        if hasattr(input_data, 'intent') and hasattr(input_data, 'context'):
            # This is intent analysis result, generate mermaid
            return await self.generate_mermaid_flowchart(input_data)
        else:
            raise ValueError("MermaidFlowGenerator expects intent analysis result")