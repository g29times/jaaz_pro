"""
🚀 COMPREHENSIVE DEMO: Text-to-Flowchart with LLM
==================================================

This is the MAIN DEMO FILE for the whiteboard processing pipeline.
It demonstrates all key features with LLM as the primary intelligence.

Usage:
    python demo.py              # Run all examples
    python demo.py --quick      # Run quick test only
    python demo.py --help       # Show options
"""

import asyncio
import json
import sys
from pathlib import Path
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType


# ============================================================================
# EXAMPLE 1: Simple Text to Flowchart
# ============================================================================

async def example_simple_text_to_flowchart():
    """Basic example: Convert text description to Mermaid flowchart"""
    print("\n" + "="*70)
    print("📝 EXAMPLE 1: Simple Text → Flowchart")
    print("="*70)

    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    pipeline = SimpleSketchToMermaidPipeline(config)

    # Simple process description
    input_data = WhiteboardInput(
        content="""
        User Login Process:
        1. User enters email and password
        2. System validates credentials
        3. If valid: Grant access to dashboard
        4. If invalid: Show error and allow retry
        """,
        input_type=InputType.TEXT
    )

    print(f"\n💬 Input: {input_data.content[:80]}...")
    print("\n🤖 Processing with LLM...")

    result = await pipeline.process_sketch_to_mermaid(input_data)

    if result.success:
        print(f"✅ Success in {result.execution_time:.2f}s")
        print(f"\n📄 Generated Mermaid:")
        print("-" * 70)
        print(result.outputs[0].content)
        print("-" * 70)

        # Show which method was used
        method = result.outputs[0].metadata.get('generation_method', 'unknown')
        print(f"\n🔧 Generation Method: {method}")

        if method == 'llm':
            print("   ✅ LLM was used as PRIMARY method!")
        else:
            print(f"   ⚠️  Fallback used: {method}")

    else:
        print(f"❌ Failed: {result.error_message}")

    return result


# ============================================================================
# EXAMPLE 2: Complex Decision Flow
# ============================================================================

async def example_complex_decision_flow():
    """Example with multiple decision points and branches"""
    print("\n" + "="*70)
    print("🔀 EXAMPLE 2: Complex Decision Flow")
    print("="*70)

    with open('config.json', 'r') as f:
        config = json.load(f)

    pipeline = SimpleSketchToMermaidPipeline(config)

    input_data = WhiteboardInput(
        content="""
        E-commerce Order Processing:
        1. Customer adds items to shopping cart
        2. Proceed to checkout
        3. Enter shipping address
        4. Select payment method
        5. Validate payment
        6. If payment succeeds:
           - Create order record
           - Send confirmation email
           - Process shipping
        7. If payment fails:
           - Show error message
           - Allow retry or cancel
        8. Update inventory
        9. Generate tracking number
        10. Notify customer of shipment
        """,
        input_type=InputType.TEXT
    )

    print(f"\n💬 Complex Process: E-commerce Order Processing")
    print("\n🤖 Processing...")

    result = await pipeline.process_sketch_to_mermaid(input_data)

    if result.success:
        output = result.outputs[0]
        print(f"✅ Success in {result.execution_time:.2f}s")

        # Count complexity
        lines = output.content.split('\n')
        node_count = output.content.count('[') + output.content.count('(')
        decision_count = output.content.count('{')

        print(f"\n📊 Flowchart Complexity:")
        print(f"   Total lines: {len(lines)}")
        print(f"   Nodes: {node_count}")
        print(f"   Decisions: {decision_count}")

        print(f"\n📄 Generated Mermaid (first 15 lines):")
        print("-" * 70)
        for line in lines[:15]:
            print(line)
        if len(lines) > 15:
            print(f"   ... ({len(lines) - 15} more lines)")
        print("-" * 70)

    return result


# ============================================================================
# EXAMPLE 3: Batch Processing
# ============================================================================

async def example_batch_processing():
    """Process multiple flowcharts in batch"""
    print("\n" + "="*70)
    print("⚡ EXAMPLE 3: Batch Processing")
    print("="*70)

    with open('config.json', 'r') as f:
        config = json.load(f)

    pipeline = SimpleSketchToMermaidPipeline(config)

    # Multiple process descriptions
    processes = [
        "User Registration: Enter email → Validate format → Send verification → Activate account",
        "Password Reset: Request reset → Send email → Enter new password → Update account",
        "API Request: Authenticate → Validate params → Process request → Return response"
    ]

    print(f"\n📦 Processing {len(processes)} flowcharts in batch...\n")

    results = []
    for i, process in enumerate(processes, 1):
        input_data = WhiteboardInput(
            content=process,
            input_type=InputType.TEXT
        )

        result = await pipeline.process_sketch_to_mermaid(input_data)

        status = "✅" if result.success else "❌"
        print(f"{status} [{i}/{len(processes)}] {process[:50]}... ({result.execution_time:.2f}s)")

        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.success)
    print(f"\n📊 Batch Results: {successful}/{len(results)} successful")

    return results


# ============================================================================
# EXAMPLE 4: Real-world Use Case
# ============================================================================

async def example_real_world_use_case():
    """Practical example: CI/CD pipeline flowchart"""
    print("\n" + "="*70)
    print("🏭 EXAMPLE 4: Real-World Use Case - CI/CD Pipeline")
    print("="*70)

    with open('config.json', 'r') as f:
        config = json.load(f)

    pipeline = SimpleSketchToMermaidPipeline(config)

    input_data = WhiteboardInput(
        content="""
        Continuous Integration/Deployment Pipeline:

        1. Developer commits code to Git repository
        2. Webhook triggers CI/CD system
        3. System pulls latest code
        4. Install dependencies
        5. Run linting and code quality checks
        6. If checks fail: Notify developer and stop
        7. If checks pass: Continue
        8. Run unit tests
        9. If tests fail: Notify developer and stop
        10. If tests pass: Build application
        11. Run integration tests
        12. If integration tests fail: Notify and stop
        13. If all tests pass: Deploy to staging environment
        14. Run smoke tests on staging
        15. If smoke tests pass: Wait for manual approval
        16. On approval: Deploy to production
        17. Monitor application health
        18. If health check fails: Trigger rollback
        19. If healthy: Complete deployment
        20. Notify team of successful deployment
        """,
        input_type=InputType.TEXT,
        metadata={"use_case": "cicd", "category": "devops"}
    )

    print("\n💬 Use Case: CI/CD Pipeline (20 steps)")
    print("\n🤖 Processing complex real-world workflow...")

    result = await pipeline.process_sketch_to_mermaid(input_data)

    if result.success:
        output = result.outputs[0]
        print(f"✅ Success in {result.execution_time:.2f}s")

        print(f"\n📄 Generated Mermaid Flowchart:")
        print("-" * 70)
        print(output.content)
        print("-" * 70)

        print(f"\n📁 Saved to: {output.file_path}")

        # Show metadata
        if output.metadata:
            print(f"\n🔍 Metadata:")
            print(f"   Method: {output.metadata.get('generation_method')}")
            print(f"   Model: {output.metadata.get('model', 'N/A')}")
            print(f"   Lines: {len(output.content.split())}")

    return result


# ============================================================================
# QUICK TEST
# ============================================================================

async def quick_test():
    """Quick test to verify system is working"""
    print("\n" + "="*70)
    print("⚡ QUICK TEST: Verify System")
    print("="*70)

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        print(f"\n⚙️  Configuration:")
        print(f"   LLM Provider: {config['mermaid_generator']['llm_provider']}")
        print(f"   Model: {config['mermaid_generator']['ollama_model']}")

        pipeline = SimpleSketchToMermaidPipeline(config)

        input_data = WhiteboardInput(
            content="Login: Enter credentials → Validate → Access granted",
            input_type=InputType.TEXT
        )

        print(f"\n💬 Test Input: {input_data.content}")
        print("\n🤖 Processing...")

        result = await pipeline.process_sketch_to_mermaid(input_data)

        if result.success:
            print(f"\n✅ System is working! ({result.execution_time:.2f}s)")
            print(f"\n📄 Output:")
            print(result.outputs[0].content)
            return True
        else:
            print(f"\n❌ Test failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

async def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("🚀 WHITEBOARD PIPELINE - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo shows the text-to-flowchart pipeline with LLM.")
    print("LLM understands requirements and generates Mermaid flowcharts.")

    examples = [
        ("Simple Text to Flowchart", example_simple_text_to_flowchart),
        ("Complex Decision Flow", example_complex_decision_flow),
        ("Batch Processing", example_batch_processing),
        ("Real-World Use Case", example_real_world_use_case),
    ]

    results = []
    for name, example_func in examples:
        try:
            result = await example_func()
            results.append((name, True, result))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            results.append((name, False, None))

    # Final summary
    print("\n" + "="*70)
    print("📊 DEMO SUMMARY")
    print("="*70)

    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\nCompleted: {successful}/{total} examples")

    for name, success, _ in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Next Steps:")
        print("  1. Try modifying the examples above")
        print("  2. Add your own process descriptions")
        print("  3. Check the generated .mmd files")
        print("  4. Ready for Phase 2: Add image input support!")
    else:
        print(f"\n⚠️  {total - successful} example(s) failed")
        print("\nTroubleshooting:")
        print("  - Make sure Ollama is running: ollama serve")
        print("  - Check model is available: ollama list")
        print("  - Verify config.json settings")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--quick', '-q']:
            # Quick test only
            asyncio.run(quick_test())
        elif sys.argv[1] in ['--help', '-h']:
            print(__doc__)
            print("\nOptions:")
            print("  --quick, -q    Run quick test only")
            print("  --help, -h     Show this help message")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for available options")
    else:
        # Run all examples
        asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()
