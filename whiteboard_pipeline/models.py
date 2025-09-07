from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import base64
from pathlib import Path


class InputType(Enum):
    SKETCH = "sketch"
    PDF = "pdf" 
    TEXT = "text"
    ARROW = "arrow"
    IMAGE = "image"


class TaskType(Enum):
    # Simplified focus on core workflow
    CREATE_FLOWCHART = "create_flowchart"
    ANALYZE_PROCESS = "analyze_process"  
    DOCUMENT_WORKFLOW = "document_workflow"


class GeneratorType(Enum):
    # Start with just Mermaid generation
    MERMAID_FLOW = "mermaid_flow"
    # Future expansions:
    # DIFFUSION_IMAGE = "diffusion_image"
    # REPORT_SUMMARIZER = "report_summarizer"


@dataclass
class WhiteboardInput:
    content: Union[str, bytes, Path]
    input_type: InputType
    metadata: Optional[Dict[str, Any]] = None
    
    def to_base64(self) -> str:
        if isinstance(self.content, (bytes, bytearray)):
            return base64.b64encode(self.content).decode('utf-8')
        elif isinstance(self.content, Path):
            with open(self.content, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return str(self.content)


@dataclass
class ParsedElement:
    element_type: str
    content: str
    confidence: float
    bbox: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParsedInput:
    elements: List[ParsedElement]
    raw_text: str
    metadata: Dict[str, Any]


@dataclass
class SemanticIntent:
    intent: str
    confidence: float
    context: Dict[str, Any]
    suggested_tasks: List[str]


@dataclass
class TaskStep:
    action: str
    generator_type: GeneratorType
    parameters: Dict[str, Any]
    dependencies: List[str] = None


@dataclass
class TaskPlan:
    steps: List[TaskStep]
    priority: int
    estimated_time: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class GeneratorOutput:
    content: Any
    output_type: str
    file_path: Optional[Path] = None
    metadata: Dict[str, Any] = None


@dataclass 
class ProcessingResult:
    outputs: List[GeneratorOutput]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    feedback_data: Optional[Dict[str, Any]] = None