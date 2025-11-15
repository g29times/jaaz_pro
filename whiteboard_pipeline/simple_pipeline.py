"""
Simplified Whiteboard Pipeline - Sketch to Mermaid Workflow

Focus on getting "Sketch → Mermaid" working first, then expand.
Following the recommendation to start small and prove the core concept.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .models import WhiteboardInput, ProcessingResult, GeneratorOutput
from .components import InputParser, VLMEngine, MermaidFlowGenerator


class SimpleSketchToMermaidPipeline:
    """Simplified pipeline focused on Sketch → Mermaid conversion"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize only the components we need for Sketch → Mermaid
        self.input_parser = InputParser(self.config.get('input_parser', {}))
        self.vlm_engine = VLMEngine(self.config.get('vlm_engine', {}))
        self.mermaid_generator = MermaidFlowGenerator(self.config.get('mermaid_generator', {}))
        
        # Feedback logging
        self.session_logs = []
        self.logger.info("SimpleSketchToMermaidPipeline initialized")
    
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