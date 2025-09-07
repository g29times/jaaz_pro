import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .models import WhiteboardInput, ProcessingResult
from .components import (
    InputParser, VLMEngine, TaskRouter, 
    GeneratorDispatcher, OutputIntegrator
)


class WhiteboardPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        self.input_parser = InputParser(self.config.get('input_parser', {}))
        self.vlm_engine = VLMEngine(self.config.get('vlm_engine', {}))
        self.task_router = TaskRouter(self.config.get('task_router', {}))
        self.generator_dispatcher = GeneratorDispatcher(self.config.get('generator_dispatcher', {}))
        self.output_integrator = OutputIntegrator(self.config.get('output_integrator', {}))
        
        self.enable_telemetry = self.config.get('enable_telemetry', True)
        self.performance_metrics = {}
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('WhiteboardPipeline')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        log_level = self.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        return logger
    
    async def process(self, whiteboard_input: WhiteboardInput) -> ProcessingResult:
        """
        Process whiteboard input through the complete pipeline
        
        Args:
            whiteboard_input: Input data to process
            
        Returns:
            ProcessingResult with all generated outputs
        """
        self.logger.info(f"Starting pipeline processing for input type: {whiteboard_input.input_type}")
        
        start_time = datetime.now()
        pipeline_context = {
            'start_time': start_time.isoformat(),
            'input_type': whiteboard_input.input_type.value,
            'pipeline_version': '0.1.0'
        }
        
        try:
            parsed_input = await self._parse_input(whiteboard_input, pipeline_context)
            
            semantic_intent = await self._extract_intent(parsed_input, pipeline_context)
            
            task_plan = await self._create_task_plan(semantic_intent, pipeline_context)
            
            generator_outputs = await self._execute_generators(task_plan, pipeline_context)
            
            final_result = await self._integrate_outputs(
                generator_outputs, whiteboard_input, pipeline_context
            )
            
            await self._record_telemetry(final_result, pipeline_context)
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return final_result
            
        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Pipeline processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                outputs=[],
                execution_time=total_time,
                success=False,
                error_message=error_msg
            )
    
    async def _parse_input(self, whiteboard_input: WhiteboardInput, context: Dict[str, Any]):
        """Step 1: Parse and extract elements from input"""
        self.logger.info("Step 1: Parsing input")
        
        step_start = datetime.now()
        
        try:
            parsed_input = await self.input_parser.parse(whiteboard_input)
            
            step_time = (datetime.now() - step_start).total_seconds()
            context['parsing_time'] = step_time
            context['elements_found'] = len(parsed_input.elements)
            
            self.logger.info(f"Input parsing completed in {step_time:.2f}s, found {len(parsed_input.elements)} elements")
            
            return parsed_input
            
        except Exception as e:
            self.logger.error(f"Input parsing failed: {e}")
            raise
    
    async def _extract_intent(self, parsed_input, context: Dict[str, Any]):
        """Step 2: Extract semantic intent using VLM"""
        self.logger.info("Step 2: Extracting semantic intent")
        
        step_start = datetime.now()
        
        try:
            semantic_intent = await self.vlm_engine.extract_intent(parsed_input)
            
            step_time = (datetime.now() - step_start).total_seconds()
            context['intent_extraction_time'] = step_time
            context['intent_confidence'] = semantic_intent.confidence
            context['intent_context'] = semantic_intent.context
            
            self.logger.info(f"Intent extraction completed in {step_time:.2f}s, intent: {semantic_intent.intent}")
            
            return semantic_intent
            
        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            raise
    
    async def _create_task_plan(self, semantic_intent, context: Dict[str, Any]):
        """Step 3: Create execution task plan"""
        self.logger.info("Step 3: Creating task plan")
        
        step_start = datetime.now()
        
        try:
            task_plan = await self.task_router.create_task_plan(semantic_intent)
            
            step_time = (datetime.now() - step_start).total_seconds()
            context['task_planning_time'] = step_time
            context['task_steps_count'] = len(task_plan.steps)
            context['task_steps'] = task_plan.steps  # Pass steps to context
            
            self.logger.info(f"Task planning completed in {step_time:.2f}s, {len(task_plan.steps)} steps planned")
            
            return task_plan
            
        except Exception as e:
            self.logger.error(f"Task planning failed: {e}")
            raise
    
    async def _execute_generators(self, task_plan, context: Dict[str, Any]):
        """Step 4: Execute generators according to task plan"""
        self.logger.info("Step 4: Executing generators")
        
        step_start = datetime.now()
        
        try:
            generator_outputs = await self.generator_dispatcher.execute_task_plan(task_plan, context)
            
            step_time = (datetime.now() - step_start).total_seconds()
            context['generation_time'] = step_time
            context['outputs_generated'] = len(generator_outputs)
            
            successful_outputs = [o for o in generator_outputs if not (o.metadata and o.metadata.get('error'))]
            self.logger.info(f"Generation completed in {step_time:.2f}s, {len(successful_outputs)}/{len(generator_outputs)} successful")
            
            return generator_outputs
            
        except Exception as e:
            self.logger.error(f"Generator execution failed: {e}")
            raise
    
    async def _integrate_outputs(self, generator_outputs, original_input, context: Dict[str, Any]):
        """Step 5: Integrate and organize outputs"""
        self.logger.info("Step 5: Integrating outputs")
        
        step_start = datetime.now()
        
        try:
            final_result = await self.output_integrator.integrate_outputs(
                generator_outputs, original_input
            )
            
            step_time = (datetime.now() - step_start).total_seconds()
            context['integration_time'] = step_time
            
            self.logger.info(f"Output integration completed in {step_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Output integration failed: {e}")
            raise
    
    async def _record_telemetry(self, result: ProcessingResult, context: Dict[str, Any]):
        """Record telemetry data for monitoring and optimization"""
        
        if not self.enable_telemetry:
            return
        
        try:
            telemetry_data = {
                'timestamp': datetime.now().isoformat(),
                'success': result.success,
                'total_execution_time': result.execution_time,
                'step_timings': {
                    'parsing': context.get('parsing_time', 0),
                    'intent_extraction': context.get('intent_extraction_time', 0),
                    'task_planning': context.get('task_planning_time', 0),
                    'generation': context.get('generation_time', 0),
                    'integration': context.get('integration_time', 0)
                },
                'metrics': {
                    'elements_found': context.get('elements_found', 0),
                    'intent_confidence': context.get('intent_confidence', 0),
                    'task_steps_count': context.get('task_steps_count', 0),
                    'outputs_generated': context.get('outputs_generated', 0)
                },
                'input_type': context.get('input_type'),
                'pipeline_version': context.get('pipeline_version')
            }
            
            if result.error_message:
                telemetry_data['error_message'] = result.error_message
            
            self.performance_metrics[datetime.now().isoformat()] = telemetry_data
            
            if len(self.performance_metrics) > 100:
                oldest_key = min(self.performance_metrics.keys())
                del self.performance_metrics[oldest_key]
            
        except Exception as e:
            self.logger.error(f"Telemetry recording failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipeline components"""
        
        self.logger.info("Performing pipeline health check")
        
        health_status = {
            'pipeline': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            health_status['components']['input_parser'] = 'healthy'
            
            health_status['components']['vlm_engine'] = 'healthy'
            
            health_status['components']['task_router'] = 'healthy'
            
            generator_tests = await self.generator_dispatcher.test_generators()
            health_status['components']['generators'] = generator_tests
            
            health_status['components']['output_integrator'] = 'healthy'
            
            all_healthy = all(
                status == 'healthy' or (isinstance(status, dict) and all(status.values()))
                for status in health_status['components'].values()
            )
            
            if not all_healthy:
                health_status['pipeline'] = 'degraded'
                
        except Exception as e:
            health_status['pipeline'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        
        if not self.performance_metrics:
            return {'message': 'No metrics available'}
        
        recent_metrics = list(self.performance_metrics.values())[-10:]
        
        avg_execution_time = sum(m['total_execution_time'] for m in recent_metrics) / len(recent_metrics)
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        
        return {
            'summary': {
                'total_sessions': len(self.performance_metrics),
                'recent_avg_execution_time': avg_execution_time,
                'recent_success_rate': success_rate
            },
            'recent_sessions': recent_metrics
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current pipeline configuration"""
        
        return {
            'pipeline_version': '0.1.0',
            'configuration': self.config,
            'components': {
                'input_parser': self.input_parser.__class__.__name__,
                'vlm_engine': self.vlm_engine.__class__.__name__,
                'task_router': self.task_router.__class__.__name__,
                'generator_dispatcher': self.generator_dispatcher.__class__.__name__,
                'output_integrator': self.output_integrator.__class__.__name__
            },
            'generator_status': self.generator_dispatcher.get_generator_status()
        }