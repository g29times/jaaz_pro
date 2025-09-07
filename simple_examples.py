"""
Simplified Examples for Sketch → Mermaid Pipeline

Following the recommendation to "start small" - focus on getting
"Sketch → Mermaid" working first, then expand.
"""

import asyncio
import json
import os
from pathlib import Path

from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType


async def load_config():
    """Load simplified configuration focused on Sketch → Mermaid"""
    
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        print("Config file not found, using minimal configuration")
        return {
            "pipeline": {"log_level": "INFO"},
            "input_parser": {"ocr_confidence_threshold": 0.3},
            "vlm_engine": {"fallback_enabled": True},
            "mermaid_generator": {"fallback_enabled": True}
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Replace environment variables
    config_str = json.dumps(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    
    return json.loads(config_str)


async def example_text_to_mermaid():
    """Core Example: Text → Mermaid Flowchart"""
    
    print("\n=== Core Example: Text → Mermaid ===")
    
    config = await load_config()
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Simple text input describing a process
    text_input = WhiteboardInput(
        content="""
        User Login Process:
        1. User enters email and password
        2. System validates credentials
        3. Check user permissions
        4. If valid, grant access
        5. If invalid, show error message
        """,
        input_type=InputType.TEXT,
        metadata={"source": "manual_input", "workflow": "authentication"}
    )
    
    print(f"Input: {text_input.content[:100]}...")
    
    result = await pipeline.process_sketch_to_mermaid(text_input)
    
    if result.success:
        print(f"✅ Success! Generated Mermaid flowchart in {result.execution_time:.2f}s")
        
        mermaid_output = result.outputs[0]
        print(f"📄 File: {mermaid_output.file_path}")
        print(f"📊 Content length: {len(mermaid_output.content)} characters")
        
        # Show first few lines of generated Mermaid
        lines = mermaid_output.content.split('\n')[:5]
        print("🔍 Generated Mermaid (first 5 lines):")
        for line in lines:
            print(f"   {line}")
        
        # Show feedback data
        if result.feedback_data:
            session_log = result.feedback_data['session_log']
            print(f"📈 Steps completed: {len(session_log['steps'])}")
            for step in session_log['steps']:
                print(f"   • {step['step']}: {step['duration']:.2f}s ({'✅' if step['success'] else '❌'})")
    else:
        print(f"❌ Failed: {result.error_message}")
    
    return result


async def example_sketch_simulation():
    """Example: Simulated Sketch → Mermaid (without actual image processing)"""
    
    print("\n=== Example: Simulated Sketch Processing ===")
    
    config = await load_config()
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Simulate what OCR might extract from a whiteboard sketch
    extracted_text = """
    START → Input Data → Validate → Decision Point
    If Valid → Process Data → Save Results → END
    If Invalid → Show Error → Input Data
    """
    
    sketch_input = WhiteboardInput(
        content=extracted_text,
        input_type=InputType.TEXT,  # Simulating OCR output as text
        metadata={
            "source": "simulated_ocr",
            "original_type": "sketch",
            "confidence": 0.85
        }
    )
    
    print(f"Simulated OCR Output: {extracted_text.strip()}")
    
    result = await pipeline.process_sketch_to_mermaid(sketch_input)
    
    if result.success:
        print(f"✅ Sketch → Mermaid conversion successful!")
        
        mermaid_content = result.outputs[0].content
        print("🎨 Generated Flowchart:")
        print(mermaid_content)
        
        # Show generation method used
        metadata = result.outputs[0].metadata
        method = metadata.get('generation_method', 'unknown')
        print(f"🔧 Generation method: {method}")
        
    else:
        print(f"❌ Conversion failed: {result.error_message}")
    
    return result


async def example_iterative_improvement():
    """Example: Multiple iterations to show feedback collection"""
    
    print("\n=== Example: Iterative Processing for Feedback ===")
    
    config = await load_config()
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Process several different workflow types
    test_workflows = [
        "Simple: Start → Process → End",
        "Complex: User Registration → Email Validation → Account Creation → Welcome Email",
        "Decision Tree: Input → Check Type → If A: Process A → End, If B: Process B → End",
        "Loop: Start → Get Data → Process → More Data? → Yes: Get Data, No: Finish"
    ]
    
    print(f"Processing {len(test_workflows)} different workflows...")
    
    for i, workflow in enumerate(test_workflows, 1):
        print(f"\n🔄 Workflow {i}: {workflow[:50]}...")
        
        input_data = WhiteboardInput(workflow, InputType.TEXT)
        result = await pipeline.process_sketch_to_mermaid(input_data)
        
        if result.success:
            print(f"   ✅ Success ({result.execution_time:.2f}s)")
        else:
            print(f"   ❌ Failed: {result.error_message}")
    
    # Show aggregated analytics
    analytics = pipeline.get_session_analytics()
    
    print(f"\n📊 Session Analytics:")
    print(f"   Total sessions: {analytics['total_sessions']}")
    print(f"   Success rate: {analytics['success_rate']:.1%}")
    print(f"   Average duration: {analytics['average_duration']:.2f}s")
    
    if 'step_performance' in analytics:
        print(f"   Step performance:")
        for step, avg_time in analytics['step_performance'].items():
            print(f"     • {step}: {avg_time:.2f}s average")
    
    return analytics


async def example_health_monitoring():
    """Example: Health check and monitoring"""
    
    print("\n=== Example: Health Monitoring ===")
    
    config = await load_config()
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Check system health
    health = await pipeline.health_check()
    
    print(f"🏥 Pipeline Health: {health['pipeline']}")
    print(f"🎯 Focus: {health['focus']}")
    print(f"📅 Checked at: {health['timestamp']}")
    
    print("🔧 Component Status:")
    for component, status in health['components'].items():
        status_icon = "✅" if status == "healthy" else "⚠️" if "fallback" in status else "❌"
        print(f"   {status_icon} {component}: {status}")
    
    # Test core functionality
    print("\n🧪 Testing core functionality...")
    
    test_input = WhiteboardInput(
        "Quick test: A → B → C",
        InputType.TEXT
    )
    
    result = await pipeline.process_sketch_to_mermaid(test_input)
    
    if result.success:
        print("✅ Core functionality test passed")
    else:
        print(f"❌ Core functionality test failed: {result.error_message}")
    
    return health


async def example_error_scenarios():
    """Example: Error handling and recovery"""
    
    print("\n=== Example: Error Handling ===")
    
    config = await load_config()
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    # Test with problematic input
    problematic_inputs = [
        WhiteboardInput("", InputType.TEXT),  # Empty input
        WhiteboardInput("Random text with no process indicators", InputType.TEXT),  # No process content
        WhiteboardInput("A" * 10000, InputType.TEXT),  # Very long input
    ]
    
    for i, input_data in enumerate(problematic_inputs, 1):
        print(f"\n🧪 Error Test {i}: {str(input_data.content)[:50]}...")
        
        result = await pipeline.process_sketch_to_mermaid(input_data)
        
        if result.success:
            print(f"   ✅ Handled gracefully - still produced output")
        else:
            print(f"   ⚠️ Failed as expected: {result.error_message[:100]}...")
        
        # Show that feedback is still collected
        if result.feedback_data:
            print("   📝 Feedback data collected for future improvement")


async def demonstration_sequence():
    """Run all examples in sequence"""
    
    print("🚀 Sketch → Mermaid Pipeline Demonstration")
    print("=" * 50)
    print("Following 'start small' philosophy - focus on core Sketch → Mermaid workflow")
    
    try:
        # Core functionality
        await example_text_to_mermaid()
        await example_sketch_simulation()
        
        # Feedback collection
        await example_iterative_improvement()
        
        # System monitoring
        await example_health_monitoring()
        
        # Error handling
        await example_error_scenarios()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        print("🎯 Core 'Sketch → Mermaid' workflow demonstrated")
        print("📊 Comprehensive logging and feedback collection shown")
        print("🔧 Production-ready with vLLM integration and mandatory OCR")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Run demonstration
    asyncio.run(demonstration_sequence())