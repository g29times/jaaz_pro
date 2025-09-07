"""
Whiteboard Processing Pipeline

A comprehensive pipeline for processing whiteboard inputs (sketches, PDFs, text, arrows)
and generating various outputs using AI/ML models.
"""

__version__ = "0.1.0"
__author__ = "Jaaz Pro Team"

from .pipeline import WhiteboardPipeline
from .models import (
    WhiteboardInput,
    ProcessingResult,
    TaskPlan,
    GeneratorOutput
)

__all__ = [
    "WhiteboardPipeline",
    "WhiteboardInput", 
    "ProcessingResult",
    "TaskPlan",
    "GeneratorOutput"
]