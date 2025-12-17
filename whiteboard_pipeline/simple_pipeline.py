"""
Simplified Whiteboard Pipeline - Sketch to Mermaid Workflow

Focus on getting "Sketch → Mermaid" working first, then expand.
Following the recommendation to start small and prove the core concept.

Now supports:
- Text → Mermaid (Phase 1) ✅
- Image → Mermaid (Phase 2) ✅
- Text → Image (Phase 3) ✅
- Combined outputs (Mermaid + Image)
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .models import WhiteboardInput, ProcessingResult, GeneratorOutput, InputType
from .components import InputParser, VLMEngine, MermaidFlowGenerator


class SimpleSketchToMermaidPipeline:
    """
    Simplified pipeline focused on Sketch → Mermaid conversion

    Now supports all 3 phases:
    - Phase 1: Text → Mermaid ✅
    - Phase 2: Image → Mermaid ✅
    - Phase 3: Text → Image ✅
    """

    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        if isinstance(config_path, str):
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file) as f:
                    self.config = json.load(f)
            else:
                self.config = {}
        else:
            self.config = config_path or {}

        self.logger = self._setup_logging()

        # Initialize components
        self.input_parser = InputParser(self.config.get('input_parser', {}))
        self.vlm_engine = VLMEngine(self.config.get('vlm_engine', {}))
        self.mermaid_generator = MermaidFlowGenerator(self.config.get('mermaid_generator', {}))

        # Phase 2: Image input handler
        try:
            from .components.image_input_handler import ImageInputHandler
            self.image_handler = ImageInputHandler(self.config.get('image_input', {}))
            self.image_handler_available = True
        except Exception as e:
            self.logger.warning(f"ImageInputHandler not available: {e}")
            self.image_handler = None
            self.image_handler_available = False

        # Phase 3: Gemini client for image generation
        try:
            from .components.gemini_client import GeminiClient
            self.gemini_client = GeminiClient(self.config.get('mermaid_generator', {}))
            self.gemini_available = True
        except Exception as e:
            self.logger.warning(f"GeminiClient not available: {e}")
            self.gemini_client = None
            self.gemini_available = False

        # Feedback logging
        self.session_logs = []
        self.logger.info("SimpleSketchToMermaidPipeline initialized")
        self.logger.info(f"  Image input support: {self.image_handler_available}")
        self.logger.info(f"  Image generation support: {self.gemini_available}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging for feedback collection"""
        logger = logging.getLogger('SimpleSketchToMermaidPipeline')
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for feedback collection
            log_file = Path(self.config.get('log_file', 'sketch_to_mermaid.log'))
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        log_level = self.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        return logger
    
    async def process_sketch_to_mermaid(self, whiteboard_input: WhiteboardInput) -> ProcessingResult:
        """
        Core workflow: Sketch → OCR → Intent → Mermaid
        
        Args:
            whiteboard_input: Input sketch or text
            
        Returns:
            ProcessingResult with Mermaid flowchart output
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"[{session_id}] Starting Sketch → Mermaid processing")
        self.logger.info(f"[{session_id}] Input type: {whiteboard_input.input_type.value}")
        
        start_time = datetime.now()
        session_log = {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'input_type': whiteboard_input.input_type.value,
            'steps': []
        }
        
        try:
            # Step 1: Parse input with mandatory OCR
            step_start = datetime.now()
            self.logger.info(f"[{session_id}] Step 1: Parsing input")
            
            parsed_input = await self.input_parser.parse(whiteboard_input)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            session_log['steps'].append({
                'step': 'input_parsing',
                'duration': step_duration,
                'success': True,
                'elements_found': len(parsed_input.elements),
                'text_length': len(parsed_input.raw_text)
            })
            
            self.logger.info(f"[{session_id}] Step 1 completed: {len(parsed_input.elements)} elements, {len(parsed_input.raw_text)} chars")
            
            # Step 2: Extract intent (should default to CREATE_FLOWCHART)
            step_start = datetime.now()
            self.logger.info(f"[{session_id}] Step 2: Extracting intent")
            
            semantic_intent = await self.vlm_engine.extract_intent(parsed_input)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            session_log['steps'].append({
                'step': 'intent_extraction',
                'duration': step_duration,
                'success': True,
                'intent': semantic_intent.intent,
                'confidence': semantic_intent.confidence
            })
            
            self.logger.info(f"[{session_id}] Step 2 completed: Intent={semantic_intent.intent}, Confidence={semantic_intent.confidence}")
            
            # Step 3: Generate Mermaid flowchart
            step_start = datetime.now()
            self.logger.info(f"[{session_id}] Step 3: Generating Mermaid flowchart")
            
            # Create task step for Mermaid generation
            from .models import TaskStep, GeneratorType
            
            task_step = TaskStep(
                action="create_mermaid_flowchart",
                generator_type=GeneratorType.MERMAID_FLOW,
                parameters={
                    "content": parsed_input.raw_text,
                    "direction": semantic_intent.context.get("flow_direction", "TD"),
                    "theme": "default",
                    "intent_context": semantic_intent.context
                }
            )
            
            context = {
                'parsed_input': parsed_input,
                'semantic_intent': semantic_intent,
                'session_id': session_id,
                'visual_elements': parsed_input.elements,  # Add visual elements for Phase 2
                'input_type': whiteboard_input.input_type.value  # Add input type for generator strategy
            }
            
            mermaid_output = await self.mermaid_generator.generate(task_step, context)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            session_log['steps'].append({
                'step': 'mermaid_generation',
                'duration': step_duration,
                'success': mermaid_output.metadata.get('error') is None,
                'output_type': mermaid_output.output_type,
                'file_created': mermaid_output.file_path is not None
            })
            
            self.logger.info(f"[{session_id}] Step 3 completed: Mermaid file created at {mermaid_output.file_path}")
            
            # Create final result
            total_duration = (datetime.now() - start_time).total_seconds()
            session_log['total_duration'] = total_duration
            session_log['success'] = True
            
            result = ProcessingResult(
                outputs=[mermaid_output],
                execution_time=total_duration,
                success=True,
                feedback_data={
                    'session_log': session_log,
                    'pipeline_type': 'sketch_to_mermaid',
                    'workflow_focus': 'core_functionality'
                }
            )
            
            self.logger.info(f"[{session_id}] Processing completed successfully in {total_duration:.2f}s")
            self._log_session_feedback(session_log)
            
            return result
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Sketch → Mermaid processing failed: {str(e)}"
            
            session_log['total_duration'] = total_duration
            session_log['success'] = False
            session_log['error'] = str(e)
            
            self.logger.error(f"[{session_id}] {error_msg}")
            self._log_session_feedback(session_log)
            
            return ProcessingResult(
                outputs=[],
                execution_time=total_duration,
                success=False,
                error_message=error_msg,
                feedback_data={
                    'session_log': session_log,
                    'pipeline_type': 'sketch_to_mermaid',
                    'error_occurred': True
                }
            )

    async def process_image_to_mermaid(self, whiteboard_input: WhiteboardInput) -> ProcessingResult:
        """
        Phase 2: Process image/sketch to Mermaid flowchart

        Args:
            whiteboard_input: Input with InputType.IMAGE and image_path

        Returns:
            ProcessingResult with Mermaid flowchart output
        """
        if not self.image_handler_available:
            return ProcessingResult(
                outputs=[],
                execution_time=0.0,
                success=False,
                error_message="ImageInputHandler not available"
            )

        if not self.gemini_available:
            return ProcessingResult(
                outputs=[],
                execution_time=0.0,
                success=False,
                error_message="GeminiClient not available"
            )

        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"[{session_id}] Starting Image → Mermaid processing")

        start_time = datetime.now()

        try:
            # Step 1: Load and preprocess image
            self.logger.info(f"[{session_id}] Loading and preprocessing image")
            image = await self.image_handler.load_and_preprocess(whiteboard_input.image_path)

            # Step 2: Generate Mermaid using Gemini Vision
            self.logger.info(f"[{session_id}] Generating Mermaid with Gemini Vision")
            flow_direction = whiteboard_input.parameters.get('direction', 'TD')

            mermaid_code = await self.gemini_client.generate_mermaid_from_image_object(
                image=image,
                flow_direction=flow_direction
            )

            if not mermaid_code:
                raise ValueError("Gemini Vision returned empty result")

            # Create output
            output = GeneratorOutput(
                content=mermaid_code,
                output_type="mermaid",
                file_path=None,
                metadata={
                    'generator': 'gemini_vision',
                    'source': 'image',
                    'image_size': self.image_handler.get_image_info(image)['size'],
                    'preprocessing': 'applied'
                }
            )

            total_duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"[{session_id}] Image → Mermaid completed in {total_duration:.2f}s")

            return ProcessingResult(
                outputs=[output],
                execution_time=total_duration,
                success=True
            )

        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Image → Mermaid processing failed: {str(e)}"
            self.logger.error(f"[{session_id}] {error_msg}")

            return ProcessingResult(
                outputs=[],
                execution_time=total_duration,
                success=False,
                error_message=error_msg
            )

    async def process_text_to_combined_output(self, whiteboard_input: WhiteboardInput,
                                             generate_image: bool = True) -> ProcessingResult:
        """
        Phase 3: Process text to BOTH Mermaid code AND visual image

        Args:
            whiteboard_input: Input with InputType.TEXT
            generate_image: Whether to also generate visual image

        Returns:
            ProcessingResult with both Mermaid code and image outputs
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"[{session_id}] Starting Text → Combined Output (Mermaid + Image)")

        start_time = datetime.now()

        try:
            # Step 1: Generate Mermaid code (existing workflow)
            self.logger.info(f"[{session_id}] Step 1: Generating Mermaid code")
            mermaid_result = await self.process_sketch_to_mermaid(whiteboard_input)

            if not mermaid_result.success:
                return mermaid_result

            mermaid_output = mermaid_result.outputs[0]
            outputs = [mermaid_output]

            # Step 2: Generate visual image (if requested and available)
            if generate_image and self.gemini_available:
                self.logger.info(f"[{session_id}] Step 2: Generating visual image")

                description = whiteboard_input.content
                style = whiteboard_input.parameters.get('image_style', 'professional flowchart diagram')

                image_bytes = await self.gemini_client.generate_diagram_image(
                    description=description,
                    style=style
                )

                if image_bytes:
                    image_output = GeneratorOutput(
                        content=image_bytes,
                        output_type="image",
                        file_path=None,
                        metadata={
                            'generator': 'gemini_image',
                            'source': 'text',
                            'format': 'png',
                            'size_bytes': len(image_bytes)
                        }
                    )
                    outputs.append(image_output)
                    self.logger.info(f"[{session_id}] Image generated: {len(image_bytes)} bytes")
                else:
                    self.logger.warning(f"[{session_id}] Image generation failed")

            total_duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(f"[{session_id}] Combined output generated in {total_duration:.2f}s")
            self.logger.info(f"[{session_id}] Outputs: {len(outputs)} ({', '.join([o.output_type for o in outputs])})")

            return ProcessingResult(
                outputs=outputs,
                execution_time=total_duration,
                success=True
            )

        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Combined output processing failed: {str(e)}"
            self.logger.error(f"[{session_id}] {error_msg}")

            return ProcessingResult(
                outputs=[],
                execution_time=total_duration,
                success=False,
                error_message=error_msg
            )

    async def process(self, whiteboard_input: WhiteboardInput,
                     generate_image: bool = False) -> ProcessingResult:
        """
        Universal processing method that routes to appropriate workflow

        Args:
            whiteboard_input: Input data
            generate_image: Whether to also generate visual image output

        Returns:
            ProcessingResult with appropriate outputs
        """
        if whiteboard_input.input_type == InputType.IMAGE:
            # Phase 2: Image → Mermaid
            return await self.process_image_to_mermaid(whiteboard_input)

        elif whiteboard_input.input_type == InputType.TEXT:
            if generate_image:
                # Phase 3: Text → Mermaid + Image
                return await self.process_text_to_combined_output(whiteboard_input, generate_image=True)
            else:
                # Phase 1: Text → Mermaid
                return await self.process_sketch_to_mermaid(whiteboard_input)

        else:
            return ProcessingResult(
                outputs=[],
                execution_time=0.0,
                success=False,
                error_message=f"Unsupported input type: {whiteboard_input.input_type}"
            )

    def _log_session_feedback(self, session_log: Dict[str, Any]):
        """Log detailed session feedback for future fine-tuning"""
        
        # Add to in-memory session logs
        self.session_logs.append(session_log)
        
        # Keep only last 100 sessions in memory
        if len(self.session_logs) > 100:
            self.session_logs = self.session_logs[-100:]
        
        # Structured logging for feedback analysis
        feedback_summary = {
            'session_id': session_log['session_id'],
            'total_duration': session_log['total_duration'],
            'success': session_log['success'],
            'steps_completed': len(session_log['steps']),
            'input_type': session_log['input_type']
        }
        
        if session_log['success']:
            self.logger.info(f"FEEDBACK_SUCCESS: {feedback_summary}")
        else:
            feedback_summary['error'] = session_log.get('error', 'unknown')
            self.logger.error(f"FEEDBACK_FAILURE: {feedback_summary}")
        
        # Log detailed step performance
        for step in session_log['steps']:
            step_feedback = {
                'session_id': session_log['session_id'],
                'step': step['step'],
                'duration': step['duration'],
                'success': step['success']
            }
            self.logger.info(f"FEEDBACK_STEP: {step_feedback}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Simplified health check for core components"""
        
        self.logger.info("Performing health check on core components")
        
        health_status = {
            'pipeline': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'focus': 'sketch_to_mermaid'
        }
        
        try:
            # Check input parser (OCR engines)
            if self.input_parser.ocr_engine or self.input_parser.backup_ocr_engine:
                health_status['components']['input_parser'] = 'healthy'
            else:
                health_status['components']['input_parser'] = 'unhealthy'
                health_status['pipeline'] = 'degraded'
            
            # Check VLM engine
            if self.vlm_engine.engine:
                health_status['components']['vlm_engine'] = 'healthy'
            else:
                health_status['components']['vlm_engine'] = 'fallback_mode'
            
            # Check Mermaid generator
            health_status['components']['mermaid_generator'] = 'healthy'
            
            # Overall status
            unhealthy_components = [k for k, v in health_status['components'].items() if v == 'unhealthy']
            if unhealthy_components:
                health_status['pipeline'] = 'unhealthy'
                health_status['unhealthy_components'] = unhealthy_components
            
        except Exception as e:
            health_status['pipeline'] = 'unhealthy'
            health_status['error'] = str(e)
        
        self.logger.info(f"Health check completed: {health_status['pipeline']}")
        return health_status
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics from recent sessions for feedback"""
        
        if not self.session_logs:
            return {'message': 'No sessions recorded yet'}
        
        successful_sessions = [s for s in self.session_logs if s['success']]
        failed_sessions = [s for s in self.session_logs if not s['success']]
        
        # Calculate averages
        if successful_sessions:
            avg_duration = sum(s['total_duration'] for s in successful_sessions) / len(successful_sessions)
            avg_steps = sum(len(s['steps']) for s in successful_sessions) / len(successful_sessions)
        else:
            avg_duration = 0
            avg_steps = 0
        
        # Step performance analysis
        step_performance = {}
        for session in successful_sessions:
            for step in session['steps']:
                step_name = step['step']
                if step_name not in step_performance:
                    step_performance[step_name] = []
                step_performance[step_name].append(step['duration'])
        
        step_averages = {
            step: sum(durations) / len(durations)
            for step, durations in step_performance.items()
        }
        
        analytics = {
            'total_sessions': len(self.session_logs),
            'successful_sessions': len(successful_sessions),
            'failed_sessions': len(failed_sessions),
            'success_rate': len(successful_sessions) / len(self.session_logs),
            'average_duration': avg_duration,
            'average_steps': avg_steps,
            'step_performance': step_averages,
            'recent_errors': [s.get('error') for s in failed_sessions[-5:] if s.get('error')],
            'focus_workflow': 'sketch_to_mermaid'
        }
        
        self.logger.info(f"Session analytics: Success rate {analytics['success_rate']:.1%}, Avg duration {avg_duration:.2f}s")
        
        return analytics
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up SimpleSketchToMermaidPipeline")
        
        if hasattr(self.vlm_engine, 'cleanup'):
            await self.vlm_engine.cleanup()
        
        self.logger.info("Cleanup completed")