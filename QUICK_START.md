# 🚀 Quick Start Guide

Get started with the Whiteboard Processing Pipeline in 5 minutes.

## Prerequisites

- Python 3.8+
- Ollama installed
- Mac or Linux

## Installation

```bash
# 1. Clone and setup
git clone <repository-url>
cd jaaz_pro

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# Or manual:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_macos.txt  # macOS
# pip install -r requirements.txt       # Linux
```

## Setup Ollama (LLM)

```bash
# 1. Install Ollama
brew install ollama  # macOS
# curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# 2. Start Ollama server
ollama serve

# 3. Model is already downloaded (qwen2.5vl:latest)
ollama list  # Verify qwen2.5vl is available
```

## Run Demo

```bash
source venv/bin/activate
python demo.py
```

That's it! You should see 4 examples running:
1. ✅ Simple text → flowchart
2. ✅ Complex decision flow
3. ✅ Batch processing
4. ✅ Real-world CI/CD pipeline

## Configuration

Edit `config.json`:

```json
{
    "mermaid_generator": {
        "llm_provider": "ollama",
        "ollama_model": "qwen2.5vl:latest",
        "temperature": 0.3
    }
}
```

## Basic Usage

```python
import asyncio
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType

async def generate_flowchart():
    pipeline = SimpleSketchToMermaidPipeline(config)

    input_data = WhiteboardInput(
        content="Login: Enter credentials → Validate → Grant access",
        input_type=InputType.TEXT
    )

    result = await pipeline.process_sketch_to_mermaid(input_data)
    print(result.outputs[0].content)

asyncio.run(generate_flowchart())
```

## Quick Test

```bash
python demo.py --quick
```

## Troubleshooting

### Issue: "Cannot connect to Ollama"

```bash
# Start Ollama
ollama serve

# Verify it's running
ollama list
```

### Issue: "Model not found"

```bash
# Check available models
ollama list

# Should see: qwen2.5vl:latest
# If not, pull it:
ollama pull qwen2.5vl
```

### Issue: Network access blocked (Claude Code)

Run in a regular terminal outside Claude Code:
```bash
cd /Users/haoxin/jaaz_pro
source venv/bin/activate
python demo.py
```

## Architecture

```
Text Input
    ↓
LLM (qwen2.5vl) - PRIMARY ⭐
├─ Semantic understanding
├─ Context-aware generation
└─ Intelligent flowchart structure
    ↓
High-Quality Mermaid Flowchart
```

**Key Point**: LLM is the PRIMARY method for text understanding, not a fallback!

## Examples

### Simple

```
Input: "Login: Enter username → Validate → Grant access"
Output: Mermaid flowchart with proper decision nodes
```

### Complex

```
Input: "E-commerce checkout process with payment validation and retry logic"
Output: Detailed flowchart with multiple decision branches
```

### Batch

```python
processes = [
    "User registration flow",
    "Password reset flow",
    "API request flow"
]
# Process all in batch
```

## Next Steps

1. ✅ Run demo: `python demo.py`
2. ✅ Try your own process descriptions
3. ✅ Check generated .mmd files
4. 🎯 Ready for Phase 2: Image sketch processing!

## Features

- ✅ **LLM-First**: Semantic understanding of requirements
- ✅ **Context-Aware**: Generates intelligent flowcharts
- ✅ **Local & Free**: Uses Ollama (no API costs)
- ✅ **Vision Ready**: qwen2.5vl supports images (Phase 2)
- ✅ **Production Ready**: Logging, fallbacks, error handling

## File Structure

```
jaaz_pro/
├── demo.py                    ← Main demo file (run this!)
├── config.json                ← Configuration
├── README.md                  ← Full documentation
├── QUICK_START.md             ← This file
└── whiteboard_pipeline/       ← Core library
    ├── simple_pipeline.py
    ├── models.py
    └── components/
        ├── ollama_client.py
        ├── generators.py
        └── ...
```

## Commands Reference

```bash
# Run all demos
python demo.py

# Quick test
python demo.py --quick

# Help
python demo.py --help

# Check Ollama
ollama list
ollama serve

# View logs
tail -f sketch_to_mermaid_feedback.log
```

## Support

- Issues: https://github.com/anthropics/claude-code/issues
- Ollama Docs: https://ollama.ai/docs
- Mermaid Syntax: https://mermaid.js.org/

## What's Next?

**Current Phase (Phase 1)**: ✅ Text → Flowchart (Working!)

**Next Phase (Phase 2)**: 🎯 Image Sketch → Flowchart
- Use qwen2.5vl's vision capabilities
- Process whiteboard photos
- Extract hand-drawn flowcharts
- Generate clean Mermaid diagrams

The vision model is already configured! Ready to build Phase 2.

---

**That's it!** You're ready to generate flowcharts from text descriptions using LLM. 🎉
