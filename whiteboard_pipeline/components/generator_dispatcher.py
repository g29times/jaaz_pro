import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..interfaces import PipelineComponent
from ..models import TaskPlan, TaskStep, GeneratorOutput, GeneratorType
from .generators import DiffusionImageGenerator, MermaidFlowGenerator, ReportSummarizer


class GeneratorDispatcher(PipelineComponent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.generators = {
            GeneratorType.DIFFUSION_IMAGE: DiffusionImageGenerator(
                config.get('diffusion_config', {})
            ),
            GeneratorType.MERMAID_FLOW: MermaidFlowGenerator(
                config.get('mermaid_config', {})
            ),
            GeneratorType.REPORT_SUMMARIZER: ReportSummarizer(
                config.get('report_config', {})
            )
        }
        
        self.max_concurrent = config.get('max_concurrent_generators', 3)
        self.timeout = config.get('generator_timeout', 300)  # 5 minutes
    
    async def execute_task_plan(self, task_plan: TaskPlan, context: Dict[str, Any]) -> List[GeneratorOutput]:
        self.logger.info(f"Executing task plan with {len(task_plan.steps)} steps")
        
        outputs = []
        execution_context = {
            **context,
            'task_plan_metadata': task_plan.metadata,
            'execution_start': datetime.now().isoformat()
        }
        
        dependency_graph = self._build_dependency_graph(task_plan.steps)
        execution_order = self._resolve_execution_order(dependency_graph)
        
        for batch in execution_order:
            batch_outputs = await self._execute_batch(batch, execution_context)
            outputs.extend(batch_outputs)
            
            self._update_context_with_outputs(execution_context, batch_outputs)
        
        return outputs
    
    def _build_dependency_graph(self, steps: List[TaskStep]) -> Dict[str, List[str]]:
        graph = {}
        
        for step in steps:
            graph[step.action] = step.dependencies or []
        
        return graph
    
    def _resolve_execution_order(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        execution_batches = []
        remaining_tasks = set(dependency_graph.keys())
        completed_tasks = set()
        
        while remaining_tasks:
            ready_tasks = []
            
            for task in remaining_tasks:
                dependencies = dependency_graph[task]
                if all(dep in completed_tasks for dep in dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                self.logger.warning("Circular dependency detected, executing remaining tasks")
                ready_tasks = list(remaining_tasks)
            
            execution_batches.append(ready_tasks)
            
            for task in ready_tasks:
                remaining_tasks.remove(task)
                completed_tasks.add(task)
        
        return execution_batches
    
    async def _execute_batch(self, task_actions: List[str], context: Dict[str, Any]) -> List[GeneratorOutput]:
        tasks = []
        
        for action in task_actions:
            task_step = self._find_task_step_by_action(action, context)
            if task_step:
                task = self._execute_single_task(task_step, context)
                tasks.append(task)
        
        if not tasks:
            return []
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                return await task
        
        limited_tasks = [limited_task(task) for task in tasks]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*limited_tasks, return_exceptions=True),
                timeout=self.timeout
            )
            
            outputs = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {task_actions[i]} failed: {result}")
                    outputs.append(self._create_error_output(task_actions[i], str(result)))
                else:
                    outputs.append(result)
            
            return outputs
            
        except asyncio.TimeoutError:
            self.logger.error(f"Batch execution timed out after {self.timeout} seconds")
            return [self._create_error_output(action, "Execution timeout") for action in task_actions]
    
    def _find_task_step_by_action(self, action: str, context: Dict[str, Any]) -> Optional[TaskStep]:
        task_plan_metadata = context.get('task_plan_metadata', {})
        
        if 'task_steps' in context:
            for step in context['task_steps']:
                if step.action == action:
                    return step
        
        return None
    
    async def _execute_single_task(self, task_step: TaskStep, context: Dict[str, Any]) -> GeneratorOutput:
        self.logger.info(f"Executing task: {task_step.action} with generator: {task_step.generator_type}")
        
        generator = self.generators.get(task_step.generator_type)
        
        if not generator:
            error_msg = f"Generator not found: {task_step.generator_type}"
            self.logger.error(error_msg)
            return self._create_error_output(task_step.action, error_msg)
        
        try:
            start_time = datetime.now()
            
            output = await generator.generate(task_step, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if output.metadata is None:
                output.metadata = {}
            
            output.metadata.update({
                'task_action': task_step.action,
                'execution_time': execution_time,
                'generator_type': task_step.generator_type.value
            })
            
            self.logger.info(f"Task {task_step.action} completed in {execution_time:.2f} seconds")
            
            return output
            
        except Exception as e:
            error_msg = f"Generator execution failed: {str(e)}"
            self.logger.error(f"Task {task_step.action} failed: {error_msg}")
            return self._create_error_output(task_step.action, error_msg)
    
    def _update_context_with_outputs(self, context: Dict[str, Any], outputs: List[GeneratorOutput]):
        if 'previous_outputs' not in context:
            context['previous_outputs'] = []
        
        context['previous_outputs'].extend(outputs)
        
        for output in outputs:
            if output.metadata and 'task_action' in output.metadata:
                action = output.metadata['task_action']
                context[f"output_{action}"] = {
                    'content': output.content,
                    'file_path': str(output.file_path) if output.file_path else None,
                    'output_type': output.output_type
                }
    
    def _create_error_output(self, action: str, error_message: str) -> GeneratorOutput:
        return GeneratorOutput(
            content=f"Error executing {action}: {error_message}",
            output_type='error',
            file_path=None,
            metadata={
                'task_action': action,
                'error': True,
                'error_message': error_message,
                'created_at': datetime.now().isoformat()
            }
        )
    
    async def process(self, input_data: Any) -> Any:
        if isinstance(input_data, TaskPlan):
            context = {'task_steps': input_data.steps}
            return await self.execute_task_plan(input_data, context)
        else:
            raise ValueError("GeneratorDispatcher expects TaskPlan as input")
    
    def get_generator_status(self) -> Dict[str, Any]:
        status = {}
        
        for generator_type, generator in self.generators.items():
            status[generator_type.value] = {
                'available': True,
                'config': generator.config,
                'class': generator.__class__.__name__
            }
        
        return status
    
    def supports_generator_type(self, generator_type: GeneratorType) -> bool:
        return generator_type in self.generators
    
    async def test_generators(self) -> Dict[str, bool]:
        results = {}
        
        for generator_type, generator in self.generators.items():
            try:
                test_step = TaskStep(
                    action="test",
                    generator_type=generator_type,
                    parameters={"test": True}
                )
                
                await asyncio.wait_for(
                    generator.generate(test_step, {"test_mode": True}),
                    timeout=30
                )
                
                results[generator_type.value] = True
                
            except Exception as e:
                self.logger.error(f"Generator {generator_type.value} test failed: {e}")
                results[generator_type.value] = False
        
        return results