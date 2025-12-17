# -*- coding: utf-8 -*-
"""
🚀 COMPREHENSIVE DEMO: Whiteboard Processing Pipeline
=====================================================

Showcases ALL 3 phases:
- Phase 1: Text → Mermaid ✅
- Phase 2: Image → Mermaid ✅
- Phase 3: Text → Image ✅
- Combined: Text → Mermaid + Image ✅

Usage:
    python demo.py              # Run all examples
    python demo.py --quick      # Run quick test only
    python demo.py --help       # Show options
"""

import asyncio
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def demo_1_text_to_mermaid():
    """
    DEMO 1: Text → Mermaid Flowchart (Phase 1)
    Classic text description to Mermaid conversion
    """
    print("\n" + "="*70)
    print("DEMO 1: Text → Mermaid Flowchart (Phase 1)")
    print("="*70 + "\n")

    from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
    from whiteboard_pipeline.models import WhiteboardInput, InputType

    pipeline = SimpleSketchToMermaidPipeline()

    # Simple CI/CD workflow
    description = """CI/CD Pipeline Workflow:
1. Developer pushes code to Git repository
2. Automated tests run
3. If tests pass: build Docker image
4. If tests fail: notify developer
5. Deploy to staging environment
6. Run integration tests
7. If integration tests pass: deploy to production
8. If integration tests fail: rollback and notify team"""

    print("📝 Input Description:")
    print(description)
    print("\n🔄 Processing...\n")

    input_data = WhiteboardInput(
        input_type=InputType.TEXT,
        content=description
    )

    result = await pipeline.process_sketch_to_mermaid(input_data)

    if result.success:
        mermaid_code = result.outputs[0].content
        print("✅ Generated Mermaid Code:")
        print("-" * 70)
        print(mermaid_code)
        print("-" * 70)
        print(f"\n⏱️  Processing time: {result.execution_time:.2f}s")

        # Save output
        output_path = Path("demo_text_to_mermaid.mmd")
        output_path.write_text(mermaid_code)
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"❌ Error: {result.error_message}")


async def demo_2_text_to_image():
    """
    DEMO 2: Text → Diagram Image (Phase 3)
    Generate visual flowchart image from text description
    """
    print("\n" + "="*70)
    print("DEMO 2: Text → Diagram Image (Phase 3)")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient
    import json

    # Load config
    with open("config.json") as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """User Authentication Flowchart:
- Start
- User enters username and password
- System validates credentials
- If invalid: show error message, allow retry (max 3 attempts)
- If max attempts reached: lock account
- If valid: check user role
- If admin: redirect to admin dashboard
- If regular user: redirect to user dashboard
- Log all login attempts
- End"""

    print("📝 Input Description:")
    print(description)
    print("\n🎨 Generating visual image...\n")

    image_bytes = await client.generate_diagram_image(
        description=description,
        style="professional flowchart with decision diamonds and clear flow"
    )

    if image_bytes:
        output_path = Path("demo_text_to_image.png")
        output_path.write_bytes(image_bytes)

        print("✅ Image Generated Successfully!")
        print(f"   Size: {len(image_bytes):,} bytes")
        print(f"💾 Saved to: {output_path}")
        print(f"\n✨ Open {output_path} to view the generated flowchart!")
    else:
        print("❌ Image generation failed")


async def demo_3_combined_output():
    """
    DEMO 3: Text → Mermaid + Image (Combined)
    Generate BOTH Mermaid code AND visual image from same description
    """
    print("\n" + "="*70)
    print("DEMO 3: Text → Mermaid + Image (Combined Output)")
    print("="*70 + "\n")

    from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
    from whiteboard_pipeline.models import WhiteboardInput, InputType

    pipeline = SimpleSketchToMermaidPipeline()

    description = """E-commerce Checkout Process:
1. Customer reviews shopping cart
2. Click 'Proceed to Checkout'
3. Enter shipping address
4. Select shipping method (Standard or Express)
5. Enter payment information
6. System processes payment
7. If payment fails: show error, offer retry
8. If payment succeeds: create order, send confirmation email
9. Display order summary page"""

    print("📝 Input Description:")
    print(description)
    print("\n🔄 Generating BOTH outputs...\n")

    input_data = WhiteboardInput(
        input_type=InputType.TEXT,
        content=description,
        parameters={'image_style': 'professional e-commerce flowchart'}
    )

    # Use the new unified process() method with generate_image=True
    result = await pipeline.process(input_data, generate_image=True)

    if result.success:
        print(f"✅ Generated {len(result.outputs)} outputs:")

        for i, output in enumerate(result.outputs, 1):
            print(f"\n   Output {i}: {output.output_type.upper()}")

            if output.output_type == "mermaid":
                mermaid_path = Path("demo_combined.mmd")
                mermaid_path.write_text(output.content)
                print(f"      📄 Mermaid code: {len(output.content)} characters")
                print(f"      💾 Saved to: {mermaid_path}")

            elif output.output_type == "image":
                image_path = Path("demo_combined.png")
                image_path.write_bytes(output.content)
                print(f"      🖼️  Image: {len(output.content):,} bytes")
                print(f"      💾 Saved to: {image_path}")

        print(f"\n⏱️  Total processing time: {result.execution_time:.2f}s")
        print(f"\n✨ Perfect! Now you have:")
        print(f"   - Mermaid code for documentation/GitHub")
        print(f"   - Visual image for presentations/slides")
    else:
        print(f"❌ Error: {result.error_message}")


async def demo_4_image_to_mermaid():
    """
    DEMO 4: Image/Sketch → Mermaid (Phase 2)
    Convert whiteboard photo or hand-drawn sketch to Mermaid
    NOTE: Requires a test image file
    """
    print("\n" + "="*70)
    print("DEMO 4: Image/Sketch → Mermaid (Phase 2)")
    print("="*70 + "\n")

    from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
    from whiteboard_pipeline.models import WhiteboardInput, InputType

    # Check if test image exists
    test_image_path = "test_images/simple_flowchart.png"

    if not Path(test_image_path).exists():
        print(f"⚠️  Skipping: Test image not found at {test_image_path}")
        print(f"\n💡 To test this feature:")
        print(f"   1. Create a test_images/ directory")
        print(f"   2. Add a flowchart image (PNG/JPG)")
        print(f"   3. Update the path in this demo")
        print(f"\n   Or use the direct Gemini Vision API:")
        print(f"   ```python")
        print(f"   from whiteboard_pipeline.components.gemini_client import GeminiClient")
        print(f"   client = GeminiClient(config)")
        print(f"   mermaid = await client.generate_mermaid_from_image('photo.jpg')")
        print(f"   ```")
        return

    pipeline = SimpleSketchToMermaidPipeline()

    print(f"📷 Input Image: {test_image_path}")
    print(f"🔄 Processing...\n")

    input_data = WhiteboardInput(
        input_type=InputType.IMAGE,
        image_path=test_image_path,
        parameters={'direction': 'TD'}
    )

    result = await pipeline.process(input_data)

    if result.success:
        mermaid_code = result.outputs[0].content
        print("✅ Generated Mermaid Code from Image:")
        print("-" * 70)
        print(mermaid_code)
        print("-" * 70)
        print(f"\n⏱️  Processing time: {result.execution_time:.2f}s")

        output_path = Path("demo_image_to_mermaid.mmd")
        output_path.write_text(mermaid_code)
        print(f"💾 Saved to: {output_path}")
    else:
        print(f"❌ Error: {result.error_message}")


async def demo_5_all_features_showcase():
    """
    DEMO 5: Complete Feature Showcase
    Demonstrate all capabilities in one workflow
    """
    print("\n" + "="*70)
    print("DEMO 5: Complete Feature Showcase")
    print("="*70 + "\n")

    from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
    from whiteboard_pipeline.models import WhiteboardInput, InputType

    pipeline = SimpleSketchToMermaidPipeline()

    descriptions = [
        {
            "name": "Simple Login",
            "desc": "User login: Enter credentials → Validate → If valid: dashboard, If invalid: error"
        },
        {
            "name": "Order Processing",
            "desc": "Order process: Receive order → Check inventory → If available: ship, If not: backorder"
        },
        {
            "name": "User Registration",
            "desc": "Registration: Fill form → Validate email → Create account → Send verification"
        }
    ]

    print(f"🚀 Processing {len(descriptions)} workflows...\n")

    for i, item in enumerate(descriptions, 1):
        print(f"[{i}/{len(descriptions)}] {item['name']}...")

        input_data = WhiteboardInput(
            input_type=InputType.TEXT,
            content=item['desc']
        )

        result = await pipeline.process(input_data, generate_image=False)

        if result.success:
            print(f"   ✅ Generated ({result.execution_time:.2f}s)")
        else:
            print(f"   ❌ Failed: {result.error_message}")

    print(f"\n✅ Batch processing complete!")


async def main():
    """Run all demos"""
    # Check for command line arguments
    if "--help" in sys.argv:
        print(__doc__)
        return

    quick_mode = "--quick" in sys.argv

    print("\n" + "="*70)
    print("🎨 COMPREHENSIVE PIPELINE DEMO")
    print("="*70)
    print("\nShowcasing all 3 phases of the Whiteboard Processing Pipeline:")
    print("  Phase 1: Text → Mermaid ✅")
    print("  Phase 2: Image → Mermaid ✅")
    print("  Phase 3: Text → Image ✅")
    print("  Plus: Combined outputs and batch processing")
    print("\n" + "="*70 + "\n")

    print("⚠️  Note: Some demos require a valid Gemini API key.")
    print("   Check config.json and ensure your key has proper permissions.\n")

    if quick_mode:
        print("🚀 QUICK MODE: Running Demo 1 only\n")
        await demo_1_text_to_mermaid()
    else:
        # Run all demos
        await demo_1_text_to_mermaid()
        await demo_2_text_to_image()
        await demo_3_combined_output()
        await demo_4_image_to_mermaid()
        await demo_5_all_features_showcase()

    print("\n" + "="*70)
    print("🎉 ALL DEMOS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_text_to_mermaid.mmd")
    if not quick_mode:
        print("  - demo_text_to_image.png")
        print("  - demo_combined.mmd")
        print("  - demo_combined.png")
        print("  - demo_image_to_mermaid.mmd (if image provided)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
