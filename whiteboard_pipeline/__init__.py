"""
Whiteboard Processing Pipeline - Sketch to Mermaid Focus

A production-ready pipeline focused on the core "Sketch → Mermaid" workflow.
Following the "start small" philosophy to perfect one workflow first.
"""

__version__ = "0.1.0"
__author__ = "Jaaz Pro Team"

from .simple_pipeline import SimpleSketchToMermaidPipeline
from .models import (
    WhiteboardInput,
    InputType,
    ProcessingResult,
    GeneratorOutput
)

__all__ = [
    "SimpleSketchToMermaidPipeline",
    "WhiteboardInput",
    "InputType", 
    "ProcessingResult",
    "GeneratorOutput"
]