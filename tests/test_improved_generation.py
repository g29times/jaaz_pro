"""
Test Improved Mermaid Generation - Phase 2
Tests the intelligent flowchart reconstruction from computer vision analysis
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType


async def test_improved_generation():
    """Test the improved intelligent Mermaid generation"""
    print("🧪 Testing Improved Mermaid Generation - Phase 2")
    print("=" * 60)
    
    # Test with the user's flowchart
    flowchart_path = Path(__file__).parent / "images" / "test_flow_chart.png"
    
    if not flowchart_path.exists():
        print(f"❌ Flowchart not found: {flowchart_path}")
        return
    
    print(f"📄 Testing with: {flowchart_path}")
    
    # Configure pipeline with intelligent analysis enabled
    config = {
        "pipeline": {"log_level": "INFO"},
        "input_parser": {
            "ocr_confidence_threshold": 0.3,
            "image_processor": {
                "min_contour_area": 100,
                "max_contour_area": 50000
            }
        },
        "vlm_engine": {"fallback_enabled": True},
        "mermaid_generator": {
            "fallback_enabled": True,
            "use_intelligent_analysis": True  # Enable intelligent analysis
        }
    }
    
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Create input
    input_data = WhiteboardInput(
        content=flowchart_path,
        input_type=InputType.IMAGE,
        metadata={"source": "improved_generation_test"}
    )
    
    print("\n🔍 Processing with Intelligent Analysis...")
    result = await pipeline.process_sketch_to_mermaid(input_data)
    
    if result.success:
        print(f"✅ Success! Processing time: {result.execution_time:.3f}s")
        
        # Show the generated Mermaid
        mermaid_output = result.outputs[0]
        print(f"\n📄 Generated Mermaid ({mermaid_output.file_path}):")
        print("=" * 40)
        print(mermaid_output.content)
        print("=" * 40)
        
        # Show generation method used
        generation_method = mermaid_output.metadata.get('generation_method', 'unknown')
        generator_type = mermaid_output.metadata.get('generator', 'unknown')
        
        print(f"\n🔧 Generation Details:")
        print(f"   Method: {generation_method}")
        print(f"   Generator: {generator_type}")
        
        if 'nodes_detected' in mermaid_output.metadata:
            print(f"   Nodes detected: {mermaid_output.metadata['nodes_detected']}")
        if 'connections_detected' in mermaid_output.metadata:
            print(f"   Connections detected: {mermaid_output.metadata['connections_detected']}")
        if 'flow_direction' in mermaid_output.metadata:
            print(f"   Flow direction: {mermaid_output.metadata['flow_direction']}")
        
        # Show session feedback
        if result.feedback_data and 'session_log' in result.feedback_data:
            session_log = result.feedback_data['session_log']
            print(f"\n📈 Pipeline Steps:")
            for step in session_log.get('steps', []):
                print(f"   • {step['step']:20}: {step['duration']:8.4f}s ({'✅' if step.get('success') else '❌'})")
        
        # Compare with original flowchart analysis
        print(f"\n📊 Visual Analysis Summary:")
        print(f"   Elements processed: {len(result.feedback_data.get('visual_elements', []))}")
        
        # Analyze the quality of the generation
        lines = mermaid_output.content.split('\n')
        node_count = len([line for line in lines if '-->' not in line and line.strip() and not line.strip().startswith('flowchart')])
        connection_count = len([line for line in lines if '-->' in line])
        
        print(f"   Generated nodes: {node_count}")
        print(f"   Generated connections: {connection_count}")
        
        print(f"\n🎯 Result Analysis:")
        if generation_method == 'computer_vision_analysis':
            print("   ✅ Used intelligent computer vision analysis!")
            print("   🎯 This should be much more accurate than fallback generation")
        elif generation_method == 'fallback':
            print("   ⚠️  Used fallback generation")
            print("   💡 Intelligent analysis may have failed - check logs")
        else:
            print(f"   ℹ️  Used method: {generation_method}")
        
    else:
        print(f"❌ Processing failed: {result.error_message}")
    
    return result


async def main():
    """Run the improved generation test"""
    result = await test_improved_generation()
    
    print(f"\n{'='*60}")
    if result and result.success:
        print("✅ Improved Mermaid generation test completed!")
        print("🎯 Check if the output is more accurate than before")
    else:
        print("❌ Test failed - check configuration and logs")


if __name__ == "__main__":
    asyncio.run(main())