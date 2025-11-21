"""
Test Google Gemini Integration

This script tests the Gemini client integration for text-to-Mermaid generation.
Before running, make sure to:
1. Add your Gemini API key to config.json
2. Install google-generativeai: pip install google-generativeai
"""

import asyncio
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_gemini_health():
    """Test if Gemini client can be initialized"""
    print("\n" + "="*60)
    print("TEST 1: Gemini Client Health Check")
    print("="*60 + "\n")

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        # Load config
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        mermaid_config = config['mermaid_generator']

        # Check if API key is set
        api_key = mermaid_config.get('gemini_api_key', '')
        if api_key == 'YOUR_GEMINI_API_KEY_HERE':
            print("⚠️  Gemini API key not configured!")
            print("   Please add your API key to config.json")
            print("   Get your key from: https://ai.google.dev/gemini-api/docs")
            return False

        # Initialize client
        client = GeminiClient(mermaid_config)
        print(f"✅ Gemini client initialized")
        print(f"   Model: {client.model_name}")

        # Health check
        health = await client.check_health()
        print(f"\n📊 Health Check Results:")
        print(f"   Status: {health['status']}")
        print(f"   API Accessible: {health.get('api_accessible', False)}")

        if health['status'] == 'healthy':
            print("\n✅ Gemini is ready to use!")
            return True
        else:
            print(f"\n❌ Health check failed: {health.get('error', 'Unknown error')}")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install required package: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_text_to_mermaid():
    """Test text-to-Mermaid generation with Gemini"""
    print("\n" + "="*60)
    print("TEST 2: Text → Mermaid Generation")
    print("="*60 + "\n")

    try:
        from whiteboard_pipeline.components.gemini_client import GeminiClient

        # Load config
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        mermaid_config = config['mermaid_generator']

        # Initialize client
        client = GeminiClient(mermaid_config)

        # Test prompt
        test_description = """
        Create a flowchart for a user login process:
        1. User enters credentials
        2. System validates credentials
        3. If valid, redirect to dashboard
        4. If invalid, show error message
        """

        print("📝 Input Description:")
        print(test_description)
        print("\n🔄 Generating Mermaid code with Gemini...\n")

        # Generate Mermaid
        mermaid_code = await client.generate_mermaid_from_text(
            test_description,
            flow_direction="TD"
        )

        if mermaid_code:
            print("✅ Mermaid Generation Successful!")
            print("\n📄 Generated Mermaid Code:")
            print("-" * 60)
            print(mermaid_code)
            print("-" * 60)

            # Save to file
            output_path = Path(__file__).parent / "test_gemini_output.mmd"
            output_path.write_text(mermaid_code)
            print(f"\n💾 Saved to: {output_path}")

            return True
        else:
            print("❌ Generation failed - returned empty result")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_to_ollama():
    """Test that Ollama fallback works when Gemini is unavailable"""
    print("\n" + "="*60)
    print("TEST 3: Gemini → Ollama Fallback")
    print("="*60 + "\n")

    try:
        from whiteboard_pipeline.components.generators import MermaidFlowGenerator
        from whiteboard_pipeline.models import TaskStep

        # Load config
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        mermaid_config = config['mermaid_generator']

        # Initialize generator (should initialize both Gemini and Ollama)
        generator = MermaidFlowGenerator(mermaid_config)

        print(f"✅ Generator initialized")
        print(f"   Gemini client: {'✅ Available' if generator.gemini_client else '❌ Not available'}")
        print(f"   Ollama client: {'✅ Available' if generator.ollama_client else '❌ Not available'}")

        # Create test task
        task = TaskStep(
            step_type='generate_flowchart',
            parameters={
                'content': 'Simple process: Start → Process → End',
                'direction': 'TD'
            }
        )

        context = {
            'session_id': 'test_session',
            'input_type': 'text',
            'visual_elements': []
        }

        print("\n🔄 Testing generation with fallback system...\n")

        # Generate (will try Gemini first, then Ollama)
        result = await generator.generate(task, context)

        if result and result.content:
            print("✅ Generation successful!")
            print(f"   Generator used: {result.metadata.get('generator', 'unknown')}")
            print(f"   Method: {result.metadata.get('generation_method', 'unknown')}")
            print("\n📄 Generated Mermaid:")
            print("-" * 60)
            print(result.content)
            print("-" * 60)
            return True
        else:
            print("❌ Generation failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 GOOGLE GEMINI INTEGRATION TESTS")
    print("="*60)

    results = {
        'health_check': False,
        'text_to_mermaid': False,
        'fallback': False
    }

    # Test 1: Health check
    results['health_check'] = await test_gemini_health()

    # Only run remaining tests if health check passes
    if results['health_check']:
        # Test 2: Text-to-Mermaid
        results['text_to_mermaid'] = await test_text_to_mermaid()

        # Test 3: Fallback system
        results['fallback'] = await test_fallback_to_ollama()
    else:
        print("\n⚠️  Skipping remaining tests due to health check failure")
        print("   Please configure your Gemini API key in config.json")

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Health Check:     {'✅ PASS' if results['health_check'] else '❌ FAIL'}")
    print(f"Text → Mermaid:   {'✅ PASS' if results['text_to_mermaid'] else '⏭️  SKIP' if not results['health_check'] else '❌ FAIL'}")
    print(f"Fallback System:  {'✅ PASS' if results['fallback'] else '⏭️  SKIP' if not results['health_check'] else '❌ FAIL'}")
    print("="*60)

    all_pass = all(results.values())
    if all_pass:
        print("\n🎉 All tests passed! Gemini integration is working correctly.")
    elif results['health_check']:
        print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n⚠️  Please configure your Gemini API key to run tests.")


if __name__ == "__main__":
    asyncio.run(main())
