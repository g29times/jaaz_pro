from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from .models import (
    WhiteboardInput, ParsedInput, SemanticIntent, 
    TaskPlan, GeneratorOutput, ProcessingResult
)


class PipelineComponent(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass


class InputParserInterface(PipelineComponent):
    @abstractmethod
    async def parse(self, whiteboard_input: WhiteboardInput) -> ParsedInput:
        pass


class VLMEngineInterface(PipelineComponent):
    @abstractmethod
    async def extract_intent(self, parsed_input: ParsedInput) -> SemanticIntent:
        pass


class TaskRouterInterface(PipelineComponent):
    @abstractmethod
    async def create_task_plan(self, intent: SemanticIntent) -> TaskPlan:
        pass


class GeneratorInterface(PipelineComponent):
    @abstractmethod
    async def generate(self, task_step: Any, context: Dict[str, Any]) -> GeneratorOutput:
        pass
    
    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        pass


class OutputIntegratorInterface(PipelineComponent):
    @abstractmethod
    async def integrate_outputs(
        self, 
        outputs: List[GeneratorOutput], 
        original_input: WhiteboardInput
    ) -> ProcessingResult:
        pass