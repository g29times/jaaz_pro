# -*- coding: utf-8 -*-
"""
Standalone Image Generation Examples
Demonstrates Gemini's native image generation capabilities for diagrams and flowcharts
Uses the "nano banana" feature - same API key as text generation!
"""

import asyncio
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


async def example_1_simple_flowchart():
    """Example 1: Simple Login Flowchart"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Login Flowchart Image")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """Simple user login flowchart:
- Start
- User enters credentials
- System validates
- If valid: redirect to dashboard
- If invalid: show error
- End"""

    print(f"📝 Description: {description}")
    print("\n🎨 Generating image...\n")

    image_bytes = await client.generate_diagram_image(
        description=description,
        style="clean flowchart with rectangles, diamonds, and arrows"
    )

    if image_bytes:
        output_path = Path(__file__).parent / "example_simple_login.png"
        output_path.write_bytes(image_bytes)
        print(f"✅ Generated: {output_path}")
    else:
        print("❌ Generation failed")


async def example_2_ecommerce_checkout():
    """Example 2: E-commerce Checkout Process"""
    print("\n" + "="*70)
    print("EXAMPLE 2: E-commerce Checkout Flowchart")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """E-commerce checkout process flowchart:
1. Customer reviews shopping cart
2. Click 'Proceed to Checkout'
3. Enter shipping address
4. Select shipping method (Standard or Express)
5. Enter payment information
6. System processes payment
7. If payment fails: show error, offer retry
8. If payment succeeds: create order
9. Send confirmation email
10. Display order summary
11. End"""

    print(f"📝 Generating e-commerce checkout flowchart...")
    print("\n🎨 Creating professional diagram...\n")

    image_bytes = await client.generate_diagram_image(
        description=description,
        style="professional flowchart diagram with clean shapes, clear labels, decision diamonds, and directional arrows"
    )

    if image_bytes:
        output_path = Path(__file__).parent / "example_ecommerce_checkout.png"
        output_path.write_bytes(image_bytes)
        print(f"✅ Generated: {output_path}")
        print(f"   Size: {len(image_bytes):,} bytes")
    else:
        print("❌ Generation failed")


async def example_3_microservices_architecture():
    """Example 3: Microservices System Architecture"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Microservices Architecture Diagram")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """Modern microservices architecture:
- Frontend layer: Web App (React), Mobile App (iOS/Android)
- API Gateway (central entry point)
- Service layer: Authentication Service, User Service, Product Service, Order Service, Payment Service
- Data layer: User Database, Product Database, Order Database, Payment Database
- Infrastructure: Redis Cache, RabbitMQ Message Queue, Elasticsearch Search Engine
- Show data flow with arrows between components
- Use boxes for services, cylinders for databases"""

    print(f"📝 Generating system architecture diagram...")
    print("\n🎨 Creating technical diagram...\n")

    image_bytes = await client.generate_diagram_image(
        description=description,
        style="clean technical architecture diagram, modern design, boxes and cylinders, clear layering"
    )

    if image_bytes:
        output_path = Path(__file__).parent / "example_microservices_arch.png"
        output_path.write_bytes(image_bytes)
        print(f"✅ Generated: {output_path}")
    else:
        print("❌ Generation failed")


async def example_4_ci_cd_pipeline():
    """Example 4: CI/CD Pipeline Diagram"""
    print("\n" + "="*70)
    print("EXAMPLE 4: CI/CD Pipeline Flowchart")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """Complete CI/CD pipeline workflow:
1. Developer pushes code to Git repository
2. Webhook triggers CI/CD pipeline
3. Run automated unit tests
4. If tests fail: notify developer, stop pipeline
5. If tests pass: build Docker image
6. Push image to container registry
7. Deploy to staging environment
8. Run integration tests on staging
9. If integration tests fail: rollback, notify team
10. If integration tests pass: wait for approval
11. Manual approval step
12. Deploy to production
13. Run smoke tests
14. If smoke tests fail: auto-rollback
15. If smoke tests pass: success, notify team
16. End"""

    print(f"📝 Generating CI/CD pipeline diagram...")
    print("\n🎨 Creating workflow visualization...\n")

    image_bytes = await client.generate_diagram_image(
        description=description,
        style="DevOps pipeline diagram with sequential steps, decision points, and clear flow direction"
    )

    if image_bytes:
        output_path = Path(__file__).parent / "example_cicd_pipeline.png"
        output_path.write_bytes(image_bytes)
        print(f"✅ Generated: {output_path}")
    else:
        print("❌ Generation failed")


async def example_5_combined_mermaid_and_image():
    """Example 5: Generate BOTH Mermaid Code and Visual Image"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Combined Output (Mermaid + Image)")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.gemini_client import GeminiClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = GeminiClient(config['mermaid_generator'])

    description = """User registration and onboarding process:
1. User visits registration page
2. Fill out registration form (name, email, password)
3. Submit form
4. System validates email format
5. If invalid: show validation error
6. If valid: check if email already exists
7. If exists: show 'email already registered' error
8. If new: create user account
9. Send verification email
10. User clicks verification link
11. Activate account
12. Redirect to welcome page
13. Show onboarding tutorial
14. End"""

    print(f"📝 Generating both Mermaid code and visual image...")
    print("\n🎨 Step 1: Generating Mermaid code...")

    # Generate Mermaid code
    mermaid_code = await client.generate_mermaid_from_text(
        description,
        flow_direction="TD"
    )

    if mermaid_code:
        mermaid_path = Path(__file__).parent / "example_combined.mmd"
        mermaid_path.write_text(mermaid_code)
        print(f"✅ Mermaid code generated: {len(mermaid_code)} characters")
        print(f"   Saved to: {mermaid_path}")
    else:
        print("❌ Mermaid generation failed")
        return

    print("\n🎨 Step 2: Generating visual image...")

    # Generate visual image
    image_bytes = await client.generate_diagram_image(
        description=description,
        style="professional onboarding flowchart with clear steps and decision points"
    )

    if image_bytes:
        image_path = Path(__file__).parent / "example_combined.png"
        image_path.write_bytes(image_bytes)
        print(f"✅ Image generated: {len(image_bytes):,} bytes")
        print(f"   Saved to: {image_path}")
    else:
        print("❌ Image generation failed")
        return

    print(f"\n🎉 Success! Generated both outputs:")
    print(f"   📄 Mermaid code: {mermaid_path}")
    print(f"   🖼️  Visual image: {image_path}")
    print(f"\n💡 Use cases:")
    print(f"   - Render Mermaid in GitHub README")
    print(f"   - Use image in presentations/slides")
    print(f"   - Both formats for different contexts")


async def main():
    """Run all image generation examples"""
    print("\n" + "="*70)
    print("🎨 GEMINI IMAGE GENERATION EXAMPLES")
    print("="*70)
    print("\nDemonstrating 'nano banana' - native image generation with Gemini!")
    print("Uses the same API key as text generation - no extra setup needed.\n")

    print("⚠️  Note: If these fail with 403 errors, verify your API key has")
    print("   access to Gemini 2.5 Flash Image at https://aistudio.google.com/\n")

    # Run all examples
    await example_1_simple_flowchart()
    await example_2_ecommerce_checkout()
    await example_3_microservices_architecture()
    await example_4_ci_cd_pipeline()
    await example_5_combined_mermaid_and_image()

    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nCheck the generated files in the current directory:")
    print("  - example_simple_login.png")
    print("  - example_ecommerce_checkout.png")
    print("  - example_microservices_arch.png")
    print("  - example_cicd_pipeline.png")
    print("  - example_combined.mmd")
    print("  - example_combined.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
