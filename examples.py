"""
Example usage of the Whiteboard Processing Pipeline

This example demonstrates how to use the pipeline to process various types of whiteboard inputs
and generate different types of outputs.
"""

import asyncio
import json
import os
from pathlib import Path

from whiteboard_pipeline import WhiteboardPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType


async def load_config():
    """Load configuration from config.json file"""
    
    config_path = Path(__file__).parent / "config.json"
    
    if not config_path.exists():
        print("Config file not found, using default configuration")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Replace environment variables in config
    config_str = json.dumps(config)
    for key, value in os.environ.items():
        config_str = config_str.replace(f"${{{key}}}", value)
    
    return json.loads(config_str)


async def example_text_processing():
    """Example: Processing text input to generate flowchart"""
    
    print("\n=== Example 1: Text to Flowchart ===")
    
    config = await load_config()
    pipeline = WhiteboardPipeline(config)
    
    text_input = WhiteboardInput(
        content="User login process: 1. Enter credentials 2. Validate user 3. Check permissions 4. Grant access or deny",
        input_type=InputType.TEXT,
        metadata={"source": "manual_input", "domain": "authentication"}
    )
    
    result = await pipeline.process(text_input)
    
    if result.success:
        print(f"✅ Processing completed in {result.execution_time:.2f} seconds")
        print(f"Generated {len(result.outputs)} outputs:")
        
        for i, output in enumerate(result.outputs, 1):
            print(f"  {i}. {output.output_type} - {output.file_path}")
    else:
        print(f"❌ Processing failed: {result.error_message}")
    
    return result


async def example_sketch_processing():
    """Example: Processing a sketch image to generate report"""
    
    print("\n=== Example 2: Sketch to Report ===")
    
    config = await load_config()
    pipeline = WhiteboardPipeline(config)
    
    # Create a mock sketch input (in real use, this would be actual image data)
    sketch_content = b"Mock sketch data - in real implementation this would be image bytes"
    
    sketch_input = WhiteboardInput(
        content=sketch_content,
        input_type=InputType.SKETCH,
        metadata={"source": "whiteboard_capture", "resolution": "1920x1080"}
    )
    
    result = await pipeline.process(sketch_input)
    
    if result.success:
        print(f"✅ Processing completed in {result.execution_time:.2f} seconds")
        print(f"Generated {len(result.outputs)} outputs:")
        
        for i, output in enumerate(result.outputs, 1):
            if output.file_path and output.file_path.exists():
                size_mb = output.file_path.stat().st_size / (1024 * 1024)
                print(f"  {i}. {output.output_type} - {output.file_path.name} ({size_mb:.2f} MB)")
            else:
                print(f"  {i}. {output.output_type} - content only")
    else:
        print(f"❌ Processing failed: {result.error_message}")
    
    return result


async def example_complex_workflow():
    """Example: Complex multi-step workflow with different output types"""
    
    print("\n=== Example 3: Complex Workflow ===")
    
    config = await load_config()
    pipeline = WhiteboardPipeline(config)
    
    complex_input = WhiteboardInput(
        content="""
        Product Development Workflow:
        
        Research Phase:
        - Market analysis
        - User interviews 
        - Competitive research
        
        Design Phase:
        - Wireframes
        - Prototypes
        - User testing
        
        Development Phase:
        - Architecture design
        - Implementation
        - Testing
        
        Launch Phase:
        - Marketing
        - Deployment
        - Monitoring
        """,
        input_type=InputType.TEXT,
        metadata={
            "source": "brainstorming_session",
            "domain": "product_management",
            "complexity": "high"
        }
    )
    
    result = await pipeline.process(complex_input)
    
    if result.success:
        print(f"✅ Processing completed in {result.execution_time:.2f} seconds")
        print(f"Generated {len(result.outputs)} outputs:")
        
        for i, output in enumerate(result.outputs, 1):
            metadata = output.metadata or {}
            generator = metadata.get('generator', 'unknown')
            print(f"  {i}. {output.output_type} (via {generator}) - {output.file_path.name if output.file_path else 'content only'}")
        
        if result.feedback_data:
            print(f"\n📊 Quality Metrics:")
            metrics = result.feedback_data.get('quality_metrics', {})
            print(f"  - Completion Rate: {metrics.get('completion_rate', 0):.1%}")
            print(f"  - Avg Generation Time: {metrics.get('average_generation_time', 0):.2f}s")
    else:
        print(f"❌ Processing failed: {result.error_message}")
    
    return result


async def example_health_check():
    """Example: Pipeline health check"""
    
    print("\n=== Example 4: Health Check ===")
    
    config = await load_config()
    pipeline = WhiteboardPipeline(config)
    
    health_status = await pipeline.health_check()
    
    print(f"Pipeline Status: {health_status['pipeline']}")
    print("Component Health:")
    
    for component, status in health_status['components'].items():
        if isinstance(status, dict):
            all_healthy = all(status.values())
            status_icon = "✅" if all_healthy else "⚠️"
            print(f"  {status_icon} {component}: {status}")
        else:
            status_icon = "✅" if status == "healthy" else "❌"
            print(f"  {status_icon} {component}: {status}")


async def example_performance_monitoring():
    """Example: Performance monitoring"""
    
    print("\n=== Example 5: Performance Monitoring ===")
    
    config = await load_config()
    pipeline = WhiteboardPipeline(config)
    
    # Process a few inputs to generate metrics
    for i in range(3):
        test_input = WhiteboardInput(
            content=f"Test input {i+1}: Quick analysis needed",
            input_type=InputType.TEXT
        )
        await pipeline.process(test_input)
    
    metrics = pipeline.get_performance_metrics()
    
    if 'summary' in metrics:
        summary = metrics['summary']
        print(f"Performance Summary:")
        print(f"  - Total Sessions: {summary['total_sessions']}")
        print(f"  - Average Execution Time: {summary['recent_avg_execution_time']:.2f}s")
        print(f"  - Success Rate: {summary['recent_success_rate']:.1%}")
    
    recent_sessions = metrics.get('recent_sessions', [])
    if recent_sessions:
        print(f"\nRecent Sessions:")
        for i, session in enumerate(recent_sessions[-3:], 1):
            status = "✅" if session['success'] else "❌"
            print(f"  {i}. {status} {session['total_execution_time']:.2f}s")


async def demonstrate_error_handling():
    """Example: Error handling and recovery"""
    
    print("\n=== Example 6: Error Handling ===")
    
    # Create config with invalid endpoints to trigger errors
    error_config = {
        "vlm_engine": {
            "vllm_endpoint": "http://invalid-endpoint:8000"
        },
        "task_router": {
            "llm_provider": "openai",
            "api_key": "invalid-key"
        }
    }
    
    pipeline = WhiteboardPipeline(error_config)
    
    error_input = WhiteboardInput(
        content="This input will likely cause errors due to invalid configuration",
        input_type=InputType.TEXT
    )
    
    result = await pipeline.process(error_input)
    
    if result.success:
        print("✅ Processing succeeded (with fallbacks)")
        print(f"Generated {len(result.outputs)} outputs using recovery mechanisms")
    else:
        print(f"❌ Processing failed: {result.error_message}")
        print("This demonstrates the pipeline's error handling capabilities")


async def main():
    """Run all examples"""
    
    print("🚀 Whiteboard Processing Pipeline Examples")
    print("=" * 50)
    
    try:
        await example_text_processing()
        await example_sketch_processing()
        await example_complex_workflow()
        await example_health_check()
        await example_performance_monitoring()
        await demonstrate_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up basic environment variables for testing
    os.environ.setdefault('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Run examples
    asyncio.run(main())