import logging
from typing import Optional, Dict, Any
from functools import wraps
import asyncio
from datetime import datetime


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    
    def __init__(self, message: str, component: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.component = component
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class InputParsingError(PipelineError):
    """Error during input parsing"""
    pass


class IntentExtractionError(PipelineError):
    """Error during intent extraction"""
    pass


class TaskPlanningError(PipelineError):
    """Error during task planning"""
    pass


class GenerationError(PipelineError):
    """Error during content generation"""
    pass


class IntegrationError(PipelineError):
    """Error during output integration"""
    pass


class ConfigurationError(PipelineError):
    """Error in pipeline configuration"""
    pass


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0, 
                      exceptions: tuple = (Exception,)):
    """Decorator for retrying functions with exponential backoff"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        break
                    
                    delay = backoff_factor * (2 ** attempt)
                    logging.getLogger(func.__module__).warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def handle_pipeline_errors(component_name: str):
    """Decorator for handling and logging pipeline component errors"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(f"{component_name}.{func.__name__}")
            
            try:
                return await func(*args, **kwargs)
            except PipelineError:
                raise
            except Exception as e:
                error_msg = f"Unexpected error in {component_name}.{func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                raise PipelineError(
                    message=error_msg,
                    component=component_name,
                    details={
                        'function': func.__name__,
                        'original_error': str(e),
                        'error_type': type(e).__name__
                    }
                )
        
        return wrapper
    return decorator


class ErrorRecoveryManager:
    """Manages error recovery strategies for pipeline components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.recovery_strategies = {
            'input_parsing': self._input_parsing_recovery,
            'intent_extraction': self._intent_extraction_recovery,
            'task_planning': self._task_planning_recovery,
            'generation': self._generation_recovery,
            'integration': self._integration_recovery
        }
    
    async def attempt_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Attempt to recover from a pipeline error"""
        
        component = error.component
        
        if component not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for component: {component}")
            return None
        
        try:
            self.logger.info(f"Attempting recovery for {component} error: {error}")
            
            recovery_result = await self.recovery_strategies[component](error, context)
            
            if recovery_result is not None:
                self.logger.info(f"Recovery successful for {component}")
                return recovery_result
            else:
                self.logger.warning(f"Recovery failed for {component}")
                return None
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for {component}: {e}")
            return None
    
    async def _input_parsing_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for input parsing errors"""
        
        try:
            from ..models import ParsedInput, ParsedElement, InputType
            
            fallback_content = context.get('raw_input_content', 'Unable to parse input content')
            
            if isinstance(fallback_content, bytes):
                fallback_content = "Binary content detected - unable to extract text"
            
            fallback_element = ParsedElement(
                element_type="text",
                content=str(fallback_content)[:500] + "...",
                confidence=0.1,
                metadata={"recovery": True, "original_error": str(error)}
            )
            
            return ParsedInput(
                elements=[fallback_element],
                raw_text=str(fallback_content)[:200],
                metadata={"recovery": True, "input_type": "fallback"}
            )
            
        except Exception as e:
            self.logger.error(f"Input parsing recovery failed: {e}")
            return None
    
    async def _intent_extraction_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for intent extraction errors"""
        
        try:
            from ..models import SemanticIntent
            
            parsed_input = context.get('parsed_input')
            
            if parsed_input and parsed_input.raw_text:
                text = parsed_input.raw_text.lower()
                
                if any(word in text for word in ['flow', 'process', 'steps', 'diagram']):
                    intent = "create_flowchart"
                    confidence = 0.4
                elif any(word in text for word in ['report', 'summary', 'document']):
                    intent = "generate_report"
                    confidence = 0.4
                elif any(word in text for word in ['image', 'picture', 'visual']):
                    intent = "generate_image"
                    confidence = 0.4
                else:
                    intent = "analyze_content"
                    confidence = 0.3
            else:
                intent = "analyze_content"
                confidence = 0.2
            
            return SemanticIntent(
                intent=intent,
                confidence=confidence,
                context={
                    "recovery": True,
                    "original_error": str(error),
                    "fallback_method": "keyword_analysis"
                },
                suggested_tasks=[intent.replace("_", " ")]
            )
            
        except Exception as e:
            self.logger.error(f"Intent extraction recovery failed: {e}")
            return None
    
    async def _task_planning_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for task planning errors"""
        
        try:
            from ..models import TaskPlan, TaskStep, GeneratorType
            
            intent = context.get('semantic_intent')
            
            if intent:
                intent_name = intent.intent
            else:
                intent_name = "analyze_content"
            
            if intent_name == "create_flowchart":
                generator_type = GeneratorType.MERMAID_FLOW
                parameters = {"content": "Create basic flowchart", "direction": "TD"}
            elif intent_name == "generate_report":
                generator_type = GeneratorType.REPORT_SUMMARIZER
                parameters = {"content": "Generate summary report", "format": "md"}
            elif intent_name == "generate_image":
                generator_type = GeneratorType.DIFFUSION_IMAGE
                parameters = {"prompt": "Abstract visualization", "style": "professional"}
            else:
                generator_type = GeneratorType.REPORT_SUMMARIZER
                parameters = {"content": "Analyze content", "format": "md", "template": "analysis"}
            
            fallback_step = TaskStep(
                action=f"fallback_{intent_name}",
                generator_type=generator_type,
                parameters=parameters
            )
            
            return TaskPlan(
                steps=[fallback_step],
                priority=3,
                estimated_time=30.0,
                metadata={
                    "recovery": True,
                    "original_error": str(error),
                    "fallback_method": "simple_task_plan"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Task planning recovery failed: {e}")
            return None
    
    async def _generation_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for generation errors"""
        
        try:
            from ..models import GeneratorOutput
            from pathlib import Path
            import tempfile
            
            error_content = f"""# Generation Error Recovery

An error occurred during content generation:

**Error:** {error}

**Component:** {error.component}

**Details:** {error.details}

**Timestamp:** {error.timestamp}

This is a fallback response to ensure pipeline completion.
"""
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = Path(tempfile.gettempdir()) / f"error_recovery_{timestamp}.md"
            error_file.write_text(error_content, encoding='utf-8')
            
            return GeneratorOutput(
                content=error_content,
                output_type='error_recovery',
                file_path=error_file,
                metadata={
                    "recovery": True,
                    "original_error": str(error),
                    "error_component": error.component,
                    "created_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Generation recovery failed: {e}")
            return None
    
    async def _integration_recovery(self, error: PipelineError, context: Dict[str, Any]) -> Optional[Any]:
        """Recovery strategy for integration errors"""
        
        try:
            from ..models import ProcessingResult, GeneratorOutput
            
            outputs = context.get('generator_outputs', [])
            
            valid_outputs = []
            for output in outputs:
                if not (output.metadata and output.metadata.get('error')):
                    valid_outputs.append(output)
            
            error_output = GeneratorOutput(
                content=f"Integration error occurred: {error}",
                output_type='error',
                metadata={
                    "recovery": True,
                    "integration_error": str(error),
                    "valid_outputs_count": len(valid_outputs)
                }
            )
            
            all_outputs = valid_outputs + [error_output]
            
            return ProcessingResult(
                outputs=all_outputs,
                execution_time=context.get('execution_time', 0),
                success=len(valid_outputs) > 0,
                error_message=f"Partial success with integration error: {error}",
                feedback_data={
                    "recovery": True,
                    "partial_success": True,
                    "valid_outputs": len(valid_outputs)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Integration recovery failed: {e}")
            return None


class CircuitBreaker:
    """Circuit breaker pattern implementation for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise PipelineError(
                    "Circuit breaker is open - service unavailable",
                    details={'failure_count': self.failure_count}
                )
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == 'half-open':
                self._reset()
            
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now().timestamp() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self):
        """Reset the circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'
        self.logger.info("Circuit breaker reset - service recovered")