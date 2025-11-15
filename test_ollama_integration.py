"""
Test script for Ollama LLM integration
Verifies that local LLM is working for Mermaid generation
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from whiteboard_pipeline.components.ollama_client import OllamaClient


async def test_ollama_health():
    """Test 1: Check Ollama server health"""
    print("\n" + "="*60)
    print("TEST 1: Ollama Server Health Check")
    print("="*60)

    config = {
        'ollama_url': 'http://localhost:11434',
        'ollama_model': 'qwen3-vl:235b-cloud',
        'temperature': 0.7
    }

    client = OllamaClient(config)
    health = await client.check_health()

    print(f"\nServer Status: {health.get('status')}")
    print(f"Server Running: {health.get('server_running')}")

    if health.get('models_available'):
        print(f"\nAvailable Models:")
        for model in health['models_available']:
            print(f"  • {model}")

    print(f"\nRequested Model: {health.get('requested_model')}")
    print(f"Model Ready: {health.get('model_ready')}")

    if health.get('error'):
        print(f"\n⚠️  Error: {health['error']}")
        return False

    if health.get('status') == 'healthy':
        print("\n✅ Ollama server is healthy and ready!")
        return True
    else:
        print("\n❌ Ollama server is not available")
        return False


async def test_simple_generation():
    """Test 2: Simple text generation"""
    print("\n" + "="*60)
    print("TEST 2: Simple Text Generation")
    print("="*60)

    config = {
        'ollama_url': 'http://localhost:11434',
        'ollama_model': 'qwen3-vl:235b-cloud',
        'temperature': 0.7,
        'timeout': 45
    }

    client = OllamaClient(config)

    prompt = "What are the three main steps in a user login process?"

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response with qwen3-vl:235b-cloud...\n")

    response = await client.generate(prompt)

    if response:
        print(f"Response ({len(response)} chars):")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print("\n✅ Simple generation successful!")
        return True
    else:
        print("\n❌ Generation failed")
        return False


async def test_mermaid_generation():
    """Test 3: Mermaid flowchart generation"""
    print("\n" + "="*60)
    print("TEST 3: Mermaid Flowchart Generation")
    print("="*60)

    config = {
        'ollama_url': 'http://localhost:11434',
        'ollama_model': 'qwen3-vl:235b-cloud',
        'temperature': 0.3,  # Lower temperature for more consistent code
        'timeout': 60
    }

    client = OllamaClient(config)

    test_description = """
    User Login Process:
    1. User enters email and password
    2. System validates credentials
    3. Check if user exists in database
    4. If valid, grant access and redirect to dashboard
    5. If invalid, show error message and retry
    """

    print(f"\nProcess Description:")
    print(test_description)
    print("\nGenerating Mermaid flowchart with qwen3-vl:235b-cloud...\n")

    mermaid_code = await client.generate_mermaid_from_text(test_description, "TD")

    if mermaid_code:
        print(f"Generated Mermaid Code ({len(mermaid_code)} chars):")
        print("-" * 60)
        print(mermaid_code)
        print("-" * 60)

        # Validate Mermaid syntax
        if 'flowchart' in mermaid_code and '-->' in mermaid_code:
            print("\n✅ Mermaid generation successful with valid syntax!")
            return True
        else:
            print("\n⚠️  Generated code may not be valid Mermaid syntax")
            return False
    else:
        print("\n❌ Mermaid generation failed")
        return False


async def test_pipeline_integration():
    """Test 4: Full pipeline integration"""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline Integration")
    print("="*60)

    from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
    from whiteboard_pipeline.models import WhiteboardInput, InputType
    import json

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    print(f"\nLLM Provider: {config['mermaid_generator']['llm_provider']}")
    print(f"Model: {config['mermaid_generator']['ollama_model']}")

    pipeline = SimpleSketchToMermaidPipeline(config)

    # Test input
    input_data = WhiteboardInput(
        content="""
        Shopping Cart Checkout:
        1. User reviews cart items
        2. Enter shipping address
        3. Select payment method
        4. Confirm order
        5. Process payment
        6. Show confirmation
        """,
        input_type=InputType.TEXT
    )

    print(f"\nProcessing: {input_data.content[:50]}...")
    print("\nGenerating flowchart with LLM...\n")

    result = await pipeline.process_sketch_to_mermaid(input_data)

    if result.success:
        print(f"✅ Pipeline Success!")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")

        if result.outputs:
            output = result.outputs[0]
            print(f"\n📄 Generated Mermaid:")
            print("-" * 60)
            print(output.content)
            print("-" * 60)
            print(f"\n📁 Saved to: {output.file_path}")

        # Check generation method
        if result.outputs[0].metadata:
            method = result.outputs[0].metadata.get('generation_method', 'unknown')
            print(f"\n🔧 Generation Method: {method}")

        return True
    else:
        print(f"\n❌ Pipeline Failed: {result.error_message}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 OLLAMA LLM INTEGRATION TEST SUITE")
    print("="*60)

    results = []

    # Test 1: Health check
    try:
        results.append(await test_ollama_health())
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        results.append(False)

    # If health check fails, skip remaining tests
    if not results[0]:
        print("\n⚠️  Ollama server not available. Please start Ollama:")
        print("   $ ollama serve")
        print("\nFor cloud model (recommended):")
        print("   $ ollama run qwen3-vl:235b-cloud")
        print("\nOr use a local model:")
        print("   $ ollama pull llama3.2")
        return

    # Test 2: Simple generation
    try:
        results.append(await test_simple_generation())
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        results.append(False)

    # Test 3: Mermaid generation
    try:
        results.append(await test_mermaid_generation())
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        results.append(False)

    # Test 4: Pipeline integration
    try:
        results.append(await test_pipeline_integration())
    except Exception as e:
        print(f"\n❌ Test 4 failed with error: {e}")
        results.append(False)

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    test_names = [
        "Ollama Server Health",
        "Simple Text Generation",
        "Mermaid Generation",
        "Pipeline Integration"
    ]

    for i, (name, result) in enumerate(zip(test_names, results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i}. {name}: {status}")

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed! LLM integration is working!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
