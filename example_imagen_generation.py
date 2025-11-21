"""
Standalone Example: Using Google Imagen for Image Generation

This example shows how to use the ImagenClient directly to generate images from text prompts.

Requirements:
1. Install google-cloud-aiplatform: pip install google-cloud-aiplatform
2. Set up Google Cloud Project with Vertex AI enabled
3. Configure service account credentials
4. Update config.json with your project_id
"""

import asyncio
import json
from pathlib import Path


async def example_1_simple_diagram():
    """Example 1: Generate a simple diagram image"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Diagram Generation")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.imagen_client import ImagenClient

    # Load config
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Initialize Imagen client
    client = ImagenClient(config['image_generator'])

    # Simple prompt
    description = "Simple flowchart showing user login process"

    print(f"Prompt: {description}")
    print("Generating image...\n")

    # Generate image
    image = await client.generate_diagram_image(
        description=description,
        style="professional diagram"
    )

    if image:
        output_path = Path(__file__).parent / "output_simple_diagram.png"
        await client.save_image(image, output_path)
        print(f"✅ Image saved to: {output_path}")
        print(f"   Size: {image.size}")
    else:
        print("❌ Generation failed")


async def example_2_flowchart_from_description():
    """Example 2: Generate flowchart from detailed description"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Flowchart from Description")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.imagen_client import ImagenClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = ImagenClient(config['image_generator'])

    # Detailed description
    description = """
    E-commerce checkout process flowchart:
    - User adds items to cart
    - Proceeds to checkout
    - Enters shipping information
    - Enters payment details
    - System validates payment
    - If payment succeeds, confirm order and send email
    - If payment fails, show error and allow retry
    """

    print(f"Description: {description}")
    print("Generating flowchart image...\n")

    # Generate flowchart-specific image
    image = await client.generate_flowchart_image(
        description=description,
        flow_direction="TD"
    )

    if image:
        output_path = Path(__file__).parent / "output_ecommerce_flowchart.png"
        await client.save_image(image, output_path)
        print(f"✅ Flowchart image saved to: {output_path}")
        print(f"   Size: {image.size}")
    else:
        print("❌ Generation failed")


async def example_3_from_mermaid_code():
    """Example 3: Generate visual image from Mermaid code"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Image from Mermaid Code")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.imagen_client import ImagenClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = ImagenClient(config['image_generator'])

    # Mermaid code
    mermaid_code = """
flowchart TD
    A([Start]) --> B[User Login]
    B --> C{Valid Credentials?}
    C -->|Yes| D[Load Dashboard]
    C -->|No| E[Show Error]
    E --> B
    D --> F([End])
    """

    print(f"Mermaid Code:\n{mermaid_code}")
    print("Generating visual representation...\n")

    # Generate image from Mermaid code
    image = await client.generate_from_mermaid(mermaid_code)

    if image:
        output_path = Path(__file__).parent / "output_from_mermaid.png"
        await client.save_image(image, output_path)
        print(f"✅ Image saved to: {output_path}")
        print(f"   Size: {image.size}")
    else:
        print("❌ Generation failed")


async def example_4_custom_style():
    """Example 4: Generate with custom style"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Style Diagram")
    print("="*70 + "\n")

    from whiteboard_pipeline.components.imagen_client import ImagenClient

    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    client = ImagenClient(config['image_generator'])

    # Description with custom style
    description = "Software deployment pipeline with multiple stages"
    style = "technical illustration, modern design, colorful"

    print(f"Description: {description}")
    print(f"Style: {style}")
    print("Generating styled diagram...\n")

    # Generate with custom style
    image = await client.generate_diagram_image(
        description=description,
        style=style
    )

    if image:
        output_path = Path(__file__).parent / "output_styled_diagram.png"
        await client.save_image(image, output_path)
        print(f"✅ Styled diagram saved to: {output_path}")
        print(f"   Size: {image.size}")
    else:
        print("❌ Generation failed")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("🎨 GOOGLE IMAGEN - IMAGE GENERATION EXAMPLES")
    print("="*70)
    print("\nThese examples show how to use ImagenClient for image generation.")
    print("\n⚠️  Requirements:")
    print("   - Google Cloud Project with Vertex AI enabled")
    print("   - Service account credentials configured")
    print("   - config.json updated with your project_id")
    print("   - Install: pip install google-cloud-aiplatform")
    print()

    # Check if configured
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    project_id = config.get('image_generator', {}).get('google_cloud_project_id', 'YOUR_GOOGLE_CLOUD_PROJECT_ID')

    if project_id == 'YOUR_GOOGLE_CLOUD_PROJECT_ID':
        print("❌ Google Cloud is not configured!")
        print("\nTo enable image generation:")
        print("1. Create a Google Cloud Project")
        print("2. Enable Vertex AI API")
        print("3. Create and download service account credentials")
        print("4. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("5. Update config.json with your project_id\n")
        return

    try:
        # Run examples
        await example_1_simple_diagram()
        await example_2_flowchart_from_description()
        await example_3_from_mermaid_code()
        await example_4_custom_style()

        print("\n" + "="*70)
        print("✅ All examples completed!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
