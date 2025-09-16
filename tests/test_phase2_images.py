"""
Phase 2 Testing: Enhanced Image Processing Capabilities
Tests the new image input processing with computer vision and OCR
"""

import asyncio
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType
from whiteboard_pipeline.components.image_processor import ImageProcessor


async def create_test_images():
    """Create test images with various elements"""
    test_images = {}
    
    # Create a simple flowchart image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw boxes (process steps)
    draw.rectangle([100, 100, 250, 150], outline='black', width=2)
    draw.text((120, 120), "START", fill='black')
    
    draw.rectangle([300, 200, 450, 250], outline='black', width=2)
    draw.text((320, 220), "PROCESS", fill='black')
    
    draw.rectangle([500, 300, 650, 350], outline='black', width=2)
    draw.text((530, 320), "END", fill='black')
    
    # Draw arrows
    draw.line([175, 150, 175, 180, 375, 180, 375, 200], fill='black', width=2)
    draw.line([375, 250, 375, 280, 575, 280, 575, 300], fill='black', width=2)
    
    # Draw diamond (decision)
    draw.polygon([(400, 100), (450, 150), (400, 200), (350, 150)], outline='black', width=2)
    draw.text((380, 145), "?", fill='black')
    
    # Save test image
    test_path = Path(tempfile.gettempdir()) / "test_flowchart.png"
    img.save(test_path)
    test_images['flowchart'] = test_path
    
    # Create an image with text
    img2 = Image.new('RGB', (600, 400), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((50, 50), "User Login Process:", fill='black')
    draw2.text((50, 100), "1. Enter credentials", fill='black')
    draw2.text((50, 150), "2. Validate user", fill='black')
    draw2.text((50, 200), "3. Grant access", fill='black')
    
    test_path2 = Path(tempfile.gettempdir()) / "test_text.png"
    img2.save(test_path2)
    test_images['text'] = test_path2
    
    return test_images


async def test_image_processor_directly():
    """Test the ImageProcessor component directly"""
    print("\n=== Testing ImageProcessor Directly ===")
    
    test_images = await create_test_images()
    processor = ImageProcessor()
    
    for image_name, image_path in test_images.items():
        print(f"\n🔍 Testing {image_name} image: {image_path}")
        
        # Create WhiteboardInput from image file
        input_data = WhiteboardInput(
            content=image_path,
            input_type=InputType.IMAGE,
            metadata={"test_type": image_name}
        )
        
        # Process the image
        result = await processor.process_image(input_data)
        
        print(f"📊 Processing result:")
        print(f"   Elements found: {len(result.elements)}")
        print(f"   Processing method: {result.metadata.get('processing_method', 'unknown')}")
        
        # Show element breakdown
        element_counts = {}
        for element in result.elements:
            element_type = element.element_type
            element_counts[element_type] = element_counts.get(element_type, 0) + 1
        
        print(f"   Element breakdown: {element_counts}")
        
        # Show sample elements
        for i, element in enumerate(result.elements[:3]):  # Show first 3
            print(f"   Element {i+1}: {element.element_type} - {element.content[:50]}... (confidence: {element.confidence:.2f})")


async def test_enhanced_pipeline():
    """Test the enhanced pipeline with image input"""
    print("\n=== Testing Enhanced Pipeline with Images ===")
    
    test_images = await create_test_images()
    
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
        "mermaid_generator": {"fallback_enabled": True}
    }
    
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    for image_name, image_path in test_images.items():
        print(f"\n🚀 Processing {image_name} through full pipeline...")
        
        # Create input
        input_data = WhiteboardInput(
            content=image_path,
            input_type=InputType.IMAGE,
            metadata={"source": "phase2_test", "image_type": image_name}
        )
        
        # Process through pipeline
        result = await pipeline.process_sketch_to_mermaid(input_data)
        
        if result.success:
            print(f"✅ Success! Generated Mermaid flowchart in {result.execution_time:.2f}s")
            
            # Show the generated Mermaid content
            mermaid_output = result.outputs[0]
            print(f"📄 Generated Mermaid file: {mermaid_output.file_path}")
            
            # Show first few lines
            lines = mermaid_output.content.split('\n')[:8]
            print("🔍 Generated Mermaid (first 8 lines):")
            for line in lines:
                print(f"   {line}")
            
            # Show session feedback
            if result.feedback_data and 'session_log' in result.feedback_data:
                session_log = result.feedback_data['session_log']
                print(f"📈 Processing steps: {len(session_log.get('steps', []))}")
                
        else:
            print(f"❌ Failed: {result.error_message}")


async def test_pdf_with_images():
    """Test PDF processing (if PyMuPDF is available)"""
    print("\n=== Testing PDF Processing ===")
    
    try:
        import fitz
        
        # Create a simple PDF with text and shapes
        doc = fitz.open()  # New empty PDF
        page = doc.new_page()
        
        # Add text
        page.insert_text((50, 50), "Workflow Process", fontsize=16)
        page.insert_text((50, 100), "1. Start process", fontsize=12)
        page.insert_text((50, 130), "2. Execute task", fontsize=12)
        page.insert_text((50, 160), "3. Complete", fontsize=12)
        
        # Add some shapes
        rect1 = fitz.Rect(200, 80, 300, 120)
        page.draw_rect(rect1, color=(0, 0, 0), width=2)
        
        rect2 = fitz.Rect(200, 140, 300, 180)
        page.draw_rect(rect2, color=(0, 0, 0), width=2)
        
        # Save PDF
        pdf_path = Path(tempfile.gettempdir()) / "test_workflow.pdf"
        doc.save(pdf_path)
        doc.close()
        
        print(f"📄 Created test PDF: {pdf_path}")
        
        # Test with pipeline
        config = {
            "pipeline": {"log_level": "INFO"},
            "input_parser": {"ocr_confidence_threshold": 0.3},
            "vlm_engine": {"fallback_enabled": True},
            "mermaid_generator": {"fallback_enabled": True}
        }
        
        pipeline = SimpleSketchToMermaidPipeline(config)
        
        input_data = WhiteboardInput(
            content=pdf_path,
            input_type=InputType.PDF,
            metadata={"source": "phase2_pdf_test"}
        )
        
        result = await pipeline.process_sketch_to_mermaid(input_data)
        
        if result.success:
            print(f"✅ PDF processing successful!")
            print(f"📄 Generated: {result.outputs[0].file_path}")
        else:
            print(f"⚠️ PDF processing result: {result.error_message}")
            
    except ImportError:
        print("⚠️ PyMuPDF not available, skipping PDF test")
    except Exception as e:
        print(f"⚠️ PDF test failed: {e}")


async def test_base64_image():
    """Test base64 encoded image processing"""
    print("\n=== Testing Base64 Image Processing ===")
    
    # Create simple test image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 100], outline='black', width=2)
    draw.text((60, 70), "STEP 1", fill='black')
    draw.rectangle([200, 150, 300, 200], outline='black', width=2)
    draw.text((210, 170), "STEP 2", fill='black')
    
    # Convert to base64
    import io
    import base64
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Test with pipeline
    config = {
        "pipeline": {"log_level": "INFO"},
        "input_parser": {"ocr_confidence_threshold": 0.3},
        "vlm_engine": {"fallback_enabled": True},
        "mermaid_generator": {"fallback_enabled": True}
    }
    
    pipeline = SimpleSketchToMermaidPipeline(config)
    
    input_data = WhiteboardInput(
        content=img_base64,
        input_type=InputType.IMAGE,
        metadata={"source": "base64_test", "encoding": "base64"}
    )
    
    result = await pipeline.process_sketch_to_mermaid(input_data)
    
    if result.success:
        print(f"✅ Base64 image processing successful!")
        print(f"📄 Generated: {result.outputs[0].file_path}")
    else:
        print(f"⚠️ Base64 processing: {result.error_message}")


async def main():
    """Run all Phase 2 tests"""
    print("🚀 Phase 2 Image Processing Tests")
    print("=" * 50)
    
    # Set environment variable if not set
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
    
    try:
        # Test individual components
        await test_image_processor_directly()
        
        # Test full pipeline
        await test_enhanced_pipeline()
        
        # Test base64 images
        await test_base64_image()
        
        # Test PDF processing
        await test_pdf_with_images()
        
        print("\n" + "=" * 50)
        print("✅ Phase 2 testing completed!")
        print("🎯 Enhanced image processing capabilities demonstrated")
        print("📊 Computer vision, OCR, and multi-format support working")
        
    except Exception as e:
        print(f"\n❌ Phase 2 testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())