"""
Test the user's actual flowchart image with Phase 2 enhanced processing
"""

import asyncio
import os
from pathlib import Path

from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType
from whiteboard_pipeline.components.image_processor import ImageProcessor


async def test_user_flowchart():
    """Test the user's provided flowchart image"""
    print("🔍 Testing User's Actual Flowchart Image")
    print("=" * 50)
    
    # Path to the user's flowchart
    flowchart_path = Path("/Users/haoxin/jaaz_pro/test_flow_chart.png")
    
    if not flowchart_path.exists():
        print(f"❌ Image not found at: {flowchart_path}")
        return
    
    print(f"📄 Found flowchart image: {flowchart_path}")
    
    # Set environment variable if not set
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
    
    # Test with ImageProcessor directly first
    print("\n=== Direct ImageProcessor Analysis ===")
    processor = ImageProcessor()
    
    input_data = WhiteboardInput(
        content=flowchart_path,
        input_type=InputType.IMAGE,
        metadata={"source": "user_provided", "description": "real_flowchart"}
    )
    
    result = await processor.process_image(input_data)
    
    print(f"📊 Processing Results:")
    print(f"   Total elements detected: {len(result.elements)}")
    print(f"   Processing method: {result.metadata.get('processing_method', 'unknown')}")
    
    # Show element breakdown
    element_counts = {}
    for element in result.elements:
        element_type = element.element_type
        element_counts[element_type] = element_counts.get(element_type, 0) + 1
    
    print(f"   Element breakdown: {element_counts}")
    
    # Show detailed elements
    print(f"\n📋 Detailed Element Analysis:")
    for i, element in enumerate(result.elements[:10]):  # Show first 10
        print(f"   {i+1:2d}. {element.element_type:12} | {element.content[:60]:60} | conf: {element.confidence:.2f}")
        if element.metadata and 'bounding_box' in element.metadata:
            bbox = element.metadata['bounding_box']
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                print(f"       └─ Location: ({bbox[0]}, {bbox[1]}) size {bbox[2]}x{bbox[3]}")
    
    if len(result.elements) > 10:
        print(f"   ... and {len(result.elements) - 10} more elements")
    
    # Show extracted text
    print(f"\n📝 Extracted Text Content:")
    text_lines = result.raw_text.split('\n')
    for i, line in enumerate(text_lines[:15]):  # Show first 15 lines
        if line.strip():
            print(f"   {i+1:2d}: {line}")
    
    if len(text_lines) > 15:
        print(f"   ... and {len(text_lines) - 15} more lines")
    
    # Test with full pipeline
    print(f"\n=== Full Pipeline Processing ===")
    
    config = {
        "pipeline": {"log_level": "INFO"},
        "input_parser": {
            "ocr_confidence_threshold": 0.3,
            "image_processor": {
                "min_contour_area": 100,
                "max_contour_area": 50000,
                "edge_threshold_low": 50,
                "edge_threshold_high": 150
            }
        },
        "vlm_engine": {"fallback_enabled": True},
        "mermaid_generator": {"fallback_enabled": True}
    }
    
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    pipeline_result = await pipeline.process_sketch_to_mermaid(input_data)
    
    if pipeline_result.success:
        print(f"✅ Pipeline processing successful!")
        print(f"⏱️  Processing time: {pipeline_result.execution_time:.3f}s")
        
        # Show generated Mermaid
        mermaid_output = pipeline_result.outputs[0]
        print(f"📄 Generated Mermaid file: {mermaid_output.file_path}")
        
        print(f"\n🎨 Generated Mermaid Code:")
        mermaid_lines = mermaid_output.content.split('\n')
        for i, line in enumerate(mermaid_lines, 1):
            print(f"   {i:2d}: {line}")
        
        # Show session feedback
        if pipeline_result.feedback_data and 'session_log' in pipeline_result.feedback_data:
            session_log = pipeline_result.feedback_data['session_log']
            print(f"\n📈 Pipeline Performance:")
            print(f"   Total steps: {len(session_log.get('steps', []))}")
            for step in session_log.get('steps', []):
                print(f"   • {step['step']:20}: {step['duration']:8.4f}s ({'✅' if step['success'] else '❌'})")
        
        # Show metadata
        if hasattr(mermaid_output, 'metadata') and mermaid_output.metadata:
            print(f"\n🔧 Generation Metadata:")
            for key, value in mermaid_output.metadata.items():
                print(f"   • {key}: {value}")
                
    else:
        print(f"❌ Pipeline processing failed: {pipeline_result.error_message}")
    
    print(f"\n" + "=" * 50)
    print(f"🎯 Real Flowchart Analysis Complete!")
    print(f"📊 This demonstrates Phase 2 capabilities on actual user content")


if __name__ == "__main__":
    asyncio.run(test_user_flowchart())