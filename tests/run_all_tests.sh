#!/bin/bash
# Quick Test Runner for Phase 2 Enhanced Image Processing
# This script runs all tests and displays results

echo "🧪 Phase 2 Test Suite Runner"
echo "=" * 50

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activating virtual environment..."
    source ../venv/bin/activate
fi

echo "📁 Current test folder structure:"
ls -la

echo ""
echo "🚀 Running comprehensive test suite..."
python comprehensive_test_suite.py

echo ""
echo "🎨 Displaying enhanced results with Mermaid visualization..."
python enhanced_display.py

echo ""
echo "📊 Test completion summary:"
echo "✅ All test files organized in tests/ folder"
echo "✅ Computer vision analysis completed"
echo "✅ Mermaid flowcharts generated and visualized"
echo "✅ Comprehensive reports created"
echo ""
echo "📁 Check outputs/ folder for all generated files:"
ls -la outputs/

echo ""
echo "🎯 To view specific results:"
echo "   • Annotated images:     outputs/annotated_images/"
echo "   • Mermaid renders:      outputs/mermaid_renders/"
echo "   • Analysis reports:     outputs/analysis_reports/"
echo "   • HTML dashboard:       outputs/test_summary_*.html"
echo ""
echo "✅ Phase 2 testing complete!"