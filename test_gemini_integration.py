# -*- coding: utf-8 -*-
"""
Comprehensive Google Gemini Integration Tests

Tests all current capabilities of the Google GenAI integration:
- API connectivity and authentication
- Text-to-Mermaid generation (simple & complex)
- Fallback system (Gemini → Ollama)
- End-to-end pipeline integration
- Performance benchmarking
"""

import asyncio
import json
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test results tracker
test_results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'tests': []
}

def record_test(name: str, passed: bool, message: str = "", duration: float = 0.0):
    """Record test result"""
    test_results['total'] += 1
    if passed:
        test_results['passed'] += 1
        status = "✅ PASS"
    else:
        test_results['failed'] += 1
        status = "❌ FAIL"

    test_results['tests'].append({
        'name': name,
        'status': status,
        'message': message,
        'duration': duration
    })

    print(f"{status} - {name} ({duration:.2f}s)")
    if message:
        print(f"       {message}")


async def test_1_api_connectivity():
    """Test 1: Verify Gemini API connectivity and authentication"""
    print("\n" + "="*70)
    print("TEST 1: API Connectivity & Authentication")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        # Load config
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        mermaid_config = config['mermaid_generator']

        # Check if API key is configured
        api_key = mermaid_config.get('gemini_api_key', '')
        if not api_key or api_key == 'YOUR_GEMINI_API_KEY_HERE':
            record_test("API Key Configuration", False, "API key not configured in config.json", time.time() - start_time)
            return False

        # Initialize client
        client = GeminiClient(mermaid_config)
        print(f"✓ Gemini client initialized")
        print(f"  Model: {client.model_name}")
        print(f"  API Key: {api_key[:20]}...")

        # Health check
        health = await client.check_health()

        if health['status'] == 'healthy':
            record_test("API Connectivity", True, f"Connected to {health.get('model', 'unknown')}", time.time() - start_time)
            print(f"\n✅ API is accessible and working!")
            return True
        else:
            error_msg = health.get('error', 'Unknown error')
            record_test("API Connectivity", False, error_msg, time.time() - start_time)
            print(f"\n❌ Health check failed: {error_msg}")
            return False

    except Exception as e:
        record_test("API Connectivity", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_2_simple_text_to_mermaid():
    """Test 2: Simple text-to-Mermaid generation"""
    print("\n" + "="*70)
    print("TEST 2: Simple Text-to-Mermaid Generation")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Simple test case
        test_description = "Simple workflow: Start, Process, End"

        print("📝 Input:")
        print(f"   {test_description}")
        print("\n🔄 Generating Mermaid code...\n")

        # Generate
        mermaid_code = await client.generate_mermaid_from_text(
            test_description,
            flow_direction="TD"
        )

        if mermaid_code:
            # Validate basic structure
            is_valid = (
                'flowchart' in mermaid_code.lower() and
                '-->' in mermaid_code
            )

            if is_valid:
                record_test("Simple Text-to-Mermaid", True, f"Generated {len(mermaid_code)} characters", time.time() - start_time)
                print("✅ Generation Successful!")
                print("\n📄 Generated Mermaid Code:")
                print("-" * 70)
                print(mermaid_code)
                print("-" * 70)

                # Save output
                output_path = Path(__file__).parent / "test_output_simple.mmd"
                output_path.write_text(mermaid_code)
                print(f"\n💾 Saved to: {output_path}")
                return True
            else:
                record_test("Simple Text-to-Mermaid", False, "Invalid Mermaid syntax", time.time() - start_time)
                print(f"❌ Invalid Mermaid syntax generated:\n{mermaid_code}")
                return False
        else:
            record_test("Simple Text-to-Mermaid", False, "Empty response", time.time() - start_time)
            print("❌ Generation returned empty result")
            return False

    except Exception as e:
        record_test("Simple Text-to-Mermaid", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_3_complex_flowchart():
    """Test 3: Complex flowchart with decisions and loops"""
    print("\n" + "="*70)
    print("TEST 3: Complex Flowchart Generation")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Complex test case
        test_description = """
        User authentication and authorization flow:
        1. User enters username and password
        2. System validates credentials
        3. If credentials are invalid, show error and allow retry (max 3 attempts)
        4. If max attempts reached, lock account
        5. If credentials are valid, check user permissions
        6. If user has admin permissions, redirect to admin dashboard
        7. If user has regular permissions, redirect to user dashboard
        8. Log all authentication attempts
        """

        print("📝 Input:")
        print(test_description)
        print("\n🔄 Generating complex Mermaid flowchart...\n")

        # Generate
        mermaid_code = await client.generate_mermaid_from_text(
            test_description,
            flow_direction="TD"
        )

        if mermaid_code:
            # Validate complex structure
            has_decisions = '?' in mermaid_code or '{' in mermaid_code or 'decision' in mermaid_code.lower()
            has_multiple_nodes = mermaid_code.count('-->') >= 5

            if has_decisions and has_multiple_nodes:
                record_test("Complex Flowchart", True, f"Generated {mermaid_code.count('-->')} connections", time.time() - start_time)
                print("✅ Complex flowchart generated successfully!")
                print("\n📄 Generated Mermaid Code:")
                print("-" * 70)
                print(mermaid_code)
                print("-" * 70)

                # Save output
                output_path = Path(__file__).parent / "test_output_complex.mmd"
                output_path.write_text(mermaid_code)
                print(f"\n💾 Saved to: {output_path}")
                return True
            else:
                record_test("Complex Flowchart", False, "Missing decision nodes or insufficient complexity", time.time() - start_time)
                print(f"⚠️ Generated flowchart may be too simple")
                print(mermaid_code)
                return False
        else:
            record_test("Complex Flowchart", False, "Empty response", time.time() - start_time)
            print("❌ Generation returned empty result")
            return False

    except Exception as e:
        record_test("Complex Flowchart", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_4_fallback_system():
    """Test 4: Test Gemini → Ollama fallback system"""
    print("\n" + "="*70)
    print("TEST 4: Fallback System (Gemini → Ollama)")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.generators import MermaidFlowGenerator
        from whiteboard_pipeline.models import TaskStep, GeneratorType

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        generator = MermaidFlowGenerator(config['mermaid_generator'])

        print(f"✓ Generator initialized")
        print(f"  Gemini client: {'✅ Available' if generator.gemini_client else '❌ Not available'}")
        print(f"  Ollama client: {'✅ Available' if generator.ollama_client else '❌ Not available'}")

        # Create test task (corrected parameters)
        task = TaskStep(
            action='generate_flowchart',
            generator_type=GeneratorType.MERMAID_FLOW,
            parameters={
                'content': 'Process: Start → Verify Input → Process Data → End',
                'direction': 'TD'
            }
        )

        context = {
            'session_id': 'test_fallback',
            'input_type': 'text',
            'visual_elements': []
        }

        print("\n🔄 Testing generation with fallback system...\n")

        # Generate (will try Gemini first, then Ollama if Gemini fails)
        result = await generator.generate(task, context)

        if result and result.content:
            generator_used = result.metadata.get('generator', 'unknown')
            method_used = result.metadata.get('generation_method', 'unknown')

            record_test("Fallback System", True, f"Used: {generator_used} ({method_used})", time.time() - start_time)
            print("✅ Generation successful!")
            print(f"  Generator used: {generator_used}")
            print(f"  Method: {method_used}")
            print(f"  Priority: {result.metadata.get('priority', 'unknown')}")
            print("\n📄 Generated Mermaid:")
            print("-" * 70)
            print(result.content)
            print("-" * 70)
            return True
        else:
            record_test("Fallback System", False, "All generation methods failed", time.time() - start_time)
            print("❌ Generation failed with all methods")
            return False

    except Exception as e:
        record_test("Fallback System", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_end_to_end_pipeline():
    """Test 5: End-to-end pipeline integration"""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Pipeline Integration")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
        from whiteboard_pipeline.models import WhiteboardInput, InputType

        # Initialize pipeline
        pipeline = SimpleSketchToMermaidPipeline()

        # Real-world test case
        test_input = WhiteboardInput(
            input_type=InputType.TEXT,
            content="""
            CI/CD Pipeline workflow:
            1. Developer pushes code to repository
            2. Automated tests run
            3. If tests pass, build Docker image
            4. If tests fail, notify developer
            5. Deploy to staging environment
            6. Run integration tests
            7. If integration tests pass, deploy to production
            8. If integration tests fail, rollback and notify team
            """
        )

        print("📝 Processing real-world use case: CI/CD Pipeline")
        print("\n🔄 Running full pipeline...\n")

        # Process through full pipeline (corrected method name)
        result = await pipeline.process_sketch_to_mermaid(test_input)

        if result.success and result.outputs:
            mermaid_code = result.outputs[0].content if result.outputs else None

            if not mermaid_code:
                record_test("End-to-End Pipeline", False, "No output generated", time.time() - start_time)
                print("❌ No Mermaid code generated")
                return False

            # Validate quality
            is_quality = (
                len(mermaid_code) > 100 and
                mermaid_code.count('-->') >= 5 and
                ('test' in mermaid_code.lower() or 'deploy' in mermaid_code.lower())
            )

            if is_quality:
                record_test("End-to-End Pipeline", True, f"Pipeline completed in {result.execution_time:.2f}s", time.time() - start_time)
                print("✅ Pipeline completed successfully!")
                print(f"\n📊 Metrics:")
                print(f"  Total duration: {result.execution_time:.2f}s")
                print(f"  Generator: {result.outputs[0].metadata.get('generator', 'unknown')}")
                print("\n📄 Generated Mermaid:")
                print("-" * 70)
                print(mermaid_code)
                print("-" * 70)

                # Save output
                output_path = Path(__file__).parent / "test_output_e2e.mmd"
                output_path.write_text(mermaid_code)
                print(f"\n💾 Saved to: {output_path}")
                return True
            else:
                record_test("End-to-End Pipeline", False, "Generated output quality too low", time.time() - start_time)
                print(f"⚠️ Output quality below threshold:\n{mermaid_code}")
                return False
        else:
            error_msg = result.error_message if result.error_message else "Unknown error"
            record_test("End-to-End Pipeline", False, error_msg, time.time() - start_time)
            print(f"❌ Pipeline failed: {error_msg}")
            return False

    except Exception as e:
        record_test("End-to-End Pipeline", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_6_performance_benchmark():
    """Test 6: Performance benchmarking"""
    print("\n" + "="*70)
    print("TEST 6: Performance Benchmarking")
    print("="*70 + "\n")

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Run multiple generations
        test_cases = [
            "Simple: Start → Process → End",
            "Login flow: Enter credentials → Validate → Redirect",
            "Shopping cart: Add item → Update total → Checkout"
        ]

        print(f"Running {len(test_cases)} generations for performance test...\n")

        timings = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] {test_case}...", end=" ")

            gen_start = time.time()
            result = await client.generate_mermaid_from_text(test_case, "TD")
            gen_time = time.time() - gen_start

            if result:
                timings.append(gen_time)
                print(f"✓ ({gen_time:.2f}s)")
            else:
                print("✗ Failed")

        if timings:
            avg_time = sum(timings) / len(timings)
            min_time = min(timings)
            max_time = max(timings)

            # Performance threshold: avg should be < 5s
            is_performant = avg_time < 5.0

            record_test("Performance Benchmark", is_performant, f"Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s", time.time() - start_time)

            print(f"\n📊 Performance Results:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Minimum: {min_time:.2f}s")
            print(f"  Maximum: {max_time:.2f}s")
            print(f"  Threshold: < 5.0s")

            if is_performant:
                print(f"\n✅ Performance is acceptable!")
                return True
            else:
                print(f"\n⚠️ Performance below threshold")
                return False
        else:
            record_test("Performance Benchmark", False, "All generations failed", time.time() - start_time)
            print("\n❌ All test generations failed")
            return False

    except Exception as e:
        record_test("Performance Benchmark", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_7_image_generation():
    """Test 7: Image generation with Gemini (native image generation)"""
    print("\n" + "="*70)
    print("TEST 7: Image Generation (Gemini Native Image)")
    print("="*70 + "\n")

    print("ℹ️  Using Gemini's native image generation:")
    print("   - Same API key as text generation")
    print("   - No additional setup required")
    print("   - Model: gemini-2.5-flash-image")
    print("   - Supports: text-to-image, image editing, style transfer")
    print()

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Initialize Gemini client (same as text generation)
        client = GeminiClient(config['mermaid_generator'])

        print(f"✓ Gemini client initialized")
        print(f"  Text model: {client.model_name}")
        print(f"  Image model: gemini-2.5-flash-image")
        print(f"  API Key: {client.api_key[:20]}...")

        # Test case: Generate a professional flowchart diagram
        test_description = """A professional flowchart diagram showing a user login process:
- Start node
- User enters username and password (process box)
- Validate credentials (decision diamond)
- If valid: redirect to dashboard (process box)
- If invalid: show error message (process box) and return to login
- End node

Use clean shapes, clear labels, arrows showing flow direction.
Style: technical diagram with sharp lines and professional appearance."""

        print(f"\n📝 Generating flowchart diagram from description")
        print("\n🔄 Calling Gemini image generation API...\n")

        # Generate image using Gemini's native image generation
        image_bytes = await client.generate_diagram_image(
            description=test_description,
            style="professional technical flowchart"
        )

        if image_bytes:
            # Save the image
            output_path = Path(__file__).parent / "test_output_gemini_image.png"

            try:
                output_path.write_bytes(image_bytes)

                # Get image info
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(image_bytes))

                record_test("Image Generation (Gemini)", True, f"Generated {img.size} image", time.time() - start_time)
                print("✅ Image generation successful!")
                print(f"\n📊 Image Details:")
                print(f"  Size: {img.size}")
                print(f"  Format: {img.format}")
                print(f"  Mode: {img.mode}")
                print(f"  File size: {len(image_bytes):,} bytes")
                print(f"\n💾 Saved to: {output_path}")
                print("\n✨ You can now open the generated image to see the flowchart!")
                print("   Note: Images include a SynthID watermark for authenticity")
                return True

            except Exception as save_error:
                record_test("Image Generation (Gemini)", False, f"Failed to save: {save_error}", time.time() - start_time)
                print(f"❌ Failed to save image: {save_error}")
                return False
        else:
            record_test("Image Generation (Gemini)", False, "No image data returned", time.time() - start_time)
            print("❌ Image generation returned empty result")
            print("\n⚠️  Possible reasons:")
            print("   - API key may not have image generation access")
            print("   - Feature may not be available in your region")
            print("   - Rate limits may be exceeded")
            return False

    except Exception as e:
        record_test("Image Generation (Gemini)", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        print("\n⚠️  Troubleshooting:")
        print("   - Ensure your API key has Gemini 2.5 access")
        print("   - Check if image generation is available in your region")
        print("   - Verify you're not hitting rate limits")
        import traceback
        traceback.print_exc()
        return False


def print_test_summary():
    """Print final test summary"""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70 + "\n")

    print(f"Total Tests:  {test_results['total']}")
    print(f"✅ Passed:    {test_results['passed']}")
    print(f"❌ Failed:    {test_results['failed']}")
    print(f"⏭️  Skipped:   {test_results['skipped']}")
    print(f"\nSuccess Rate: {(test_results['passed'] / test_results['total'] * 100) if test_results['total'] > 0 else 0:.1f}%")

    print("\n" + "-"*70)
    print("Detailed Results:")
    print("-"*70)
    for test in test_results['tests']:
        print(f"{test['status']} - {test['name']} ({test['duration']:.2f}s)")
        if test['message']:
            print(f"         {test['message']}")

    print("\n" + "="*70)

    if test_results['failed'] == 0:
        print("🎉 ALL TESTS PASSED!")
        print("Google Gemini integration is working perfectly!")
    else:
        print(f"⚠️  {test_results['failed']} test(s) failed. Please review the errors above.")

    print("="*70 + "\n")


async def test_8_flowchart_image_generation():
    """Test 8: Generate flowchart diagram image from text description"""
    print("\n" + "="*70)
    print("TEST 8: Flowchart Diagram Image Generation")
    print("="*70 + "\n")

    print("ℹ️  Testing specialized flowchart image generation")
    print()

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Test case: Generate a professional flowchart diagram
        test_description = """E-commerce checkout process:
- User reviews cart items
- Click checkout button
- Enter shipping address
- Choose shipping method (Standard or Express)
- Enter payment information
- System validates payment
- If payment fails: show error, allow retry
- If payment succeeds: create order, send confirmation email
- Display order confirmation page"""

        print(f"📝 Generating professional flowchart image")
        print(f"   Description: E-commerce checkout process")
        print("\n🔄 Generating with Gemini Flash Image...\n")

        # Generate flowchart-style image
        image_bytes = await client.generate_diagram_image(
            description=test_description,
            style="professional flowchart diagram with clean shapes, clear labels, arrows, and decision diamonds"
        )

        if image_bytes:
            output_path = Path(__file__).parent / "test_output_flowchart.png"

            try:
                output_path.write_bytes(image_bytes)

                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_bytes))

                record_test("Flowchart Image Generation", True, f"Generated {img.size} flowchart", time.time() - start_time)
                print("✅ Flowchart image generation successful!")
                print(f"\n📊 Image Details:")
                print(f"  Size: {img.size}")
                print(f"  File: {output_path.name}")
                print(f"  Type: Professional flowchart diagram")
                print(f"\n💾 Saved to: {output_path}")
                return True

            except Exception as save_error:
                record_test("Flowchart Image Generation", False, f"Save failed: {save_error}", time.time() - start_time)
                print(f"❌ Failed to save: {save_error}")
                return False
        else:
            record_test("Flowchart Image Generation", False, "No image returned", time.time() - start_time)
            print("❌ Image generation returned empty result")
            return False

    except Exception as e:
        record_test("Flowchart Image Generation", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_9_technical_diagram_generation():
    """Test 9: Generate technical system architecture diagram"""
    print("\n" + "="*70)
    print("TEST 9: Technical Architecture Diagram Generation")
    print("="*70 + "\n")

    print("ℹ️  Testing technical diagram generation")
    print()

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Test case: System architecture diagram
        test_description = """Microservices architecture diagram:
- Frontend (React app) at the top
- API Gateway in the middle
- Three backend services: User Service, Order Service, Payment Service
- Databases: User DB, Order DB, Payment DB (one for each service)
- Message Queue connecting all services
- Show arrows indicating data flow between components"""

        print(f"📝 Generating technical architecture diagram")
        print(f"   Type: Microservices architecture")
        print("\n🔄 Generating...\n")

        # Generate technical diagram
        image_bytes = await client.generate_diagram_image(
            description=test_description,
            style="clean technical architecture diagram, modern design, boxes and arrows"
        )

        if image_bytes:
            output_path = Path(__file__).parent / "test_output_architecture.png"

            try:
                output_path.write_bytes(image_bytes)

                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_bytes))

                record_test("Technical Diagram Generation", True, f"Generated {img.size} diagram", time.time() - start_time)
                print("✅ Technical diagram generation successful!")
                print(f"\n📊 Image Details:")
                print(f"  Size: {img.size}")
                print(f"  Type: System architecture diagram")
                print(f"\n💾 Saved to: {output_path}")
                return True

            except Exception as save_error:
                record_test("Technical Diagram Generation", False, f"Save failed: {save_error}", time.time() - start_time)
                print(f"❌ Failed to save: {save_error}")
                return False
        else:
            record_test("Technical Diagram Generation", False, "No image returned", time.time() - start_time)
            print("❌ Image generation returned empty result")
            return False

    except Exception as e:
        record_test("Technical Diagram Generation", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_10_combined_mermaid_and_image():
    """Test 10: Generate both Mermaid code AND image from same description"""
    print("\n" + "="*70)
    print("TEST 10: Combined Mermaid + Image Generation")
    print("="*70 + "\n")

    print("ℹ️  Testing combined output (Mermaid code + visual image)")
    print()

    start_time = time.time()

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        client = GeminiClient(config['mermaid_generator'])

        # Test case: User authentication flow
        test_description = """User login and authentication process:
1. User enters username and password
2. System validates credentials
3. If invalid: show error message, allow retry (max 3 attempts)
4. If max attempts: lock account for 30 minutes
5. If valid: check user role
6. If admin: redirect to admin dashboard
7. If regular user: redirect to user dashboard
8. Log all login attempts"""

        print(f"📝 Generating BOTH Mermaid code and visual image")
        print(f"   Process: User authentication flow")
        print("\n🔄 Step 1: Generating Mermaid code...")

        # Generate Mermaid code
        mermaid_code = await client.generate_mermaid_from_text(
            test_description,
            flow_direction="TD"
        )

        print("✓ Mermaid code generated")
        print("\n🔄 Step 2: Generating visual image...")

        # Generate visual image
        image_bytes = await client.generate_diagram_image(
            description=test_description,
            style="professional flowchart with decision diamonds and clear flow"
        )

        if mermaid_code and image_bytes:
            # Save Mermaid code
            mermaid_path = Path(__file__).parent / "test_output_combined.mmd"
            mermaid_path.write_text(mermaid_code)

            # Save image
            image_path = Path(__file__).parent / "test_output_combined.png"
            image_path.write_bytes(image_bytes)

            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))

            record_test("Combined Output", True, "Both Mermaid and image generated", time.time() - start_time)
            print("✅ Combined generation successful!")
            print(f"\n📊 Outputs:")
            print(f"  Mermaid code: {len(mermaid_code)} characters")
            print(f"  Image size: {img.size}")
            print(f"\n💾 Saved files:")
            print(f"  {mermaid_path}")
            print(f"  {image_path}")
            print(f"\n✨ You can now:")
            print(f"  - Render Mermaid code in documentation")
            print(f"  - Use image in presentations")
            return True

        else:
            failed_part = "Mermaid code" if not mermaid_code else "Image"
            record_test("Combined Output", False, f"{failed_part} generation failed", time.time() - start_time)
            print(f"❌ {failed_part} generation failed")
            return False

    except Exception as e:
        record_test("Combined Output", False, str(e), time.time() - start_time)
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE GOOGLE GEMINI INTEGRATION TESTS")
    print("="*70)
    print("\nTesting all current capabilities:")
    print("  1. API Connectivity")
    print("  2. Simple Text-to-Mermaid")
    print("  3. Complex Flowchart Generation")
    print("  4. Fallback System")
    print("  5. End-to-End Pipeline")
    print("  6. Performance Benchmark")
    print("  7. Image Generation (Basic)")
    print("  8. Flowchart Image Generation")
    print("  9. Technical Diagram Generation")
    print(" 10. Combined Mermaid + Image Output")

    # Run all tests
    await test_1_api_connectivity()
    await test_2_simple_text_to_mermaid()
    await test_3_complex_flowchart()
    await test_4_fallback_system()
    await test_5_end_to_end_pipeline()
    await test_6_performance_benchmark()
    await test_7_image_generation()
    await test_8_flowchart_image_generation()
    await test_9_technical_diagram_generation()
    await test_10_combined_mermaid_and_image()

    # Print summary
    print_test_summary()

    # Exit code (only count non-skipped failures)
    actual_failures = test_results['failed'] - test_results['skipped']
    exit_code = 0 if actual_failures == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
