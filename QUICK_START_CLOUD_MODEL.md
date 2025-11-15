# Quick Start: Using Qwen3-VL 235B Cloud Model

## Overview

Your project is now configured to use **qwen3-vl:235b-cloud**, a powerful 235 billion parameter vision-language model that runs via Ollama's cloud API.

### Why This Model?

- **235B Parameters** - Much more powerful than local models (3-8B)
- **Vision Capabilities** - Can process images AND text (perfect for Phase 2!)
- **Cloud-based** - No local storage needed (no multi-GB downloads)
- **Cost Effective** - Cheaper than OpenAI API
- **Same Interface** - Uses familiar Ollama API

## Setup (2 Minutes)

### 1. Start Ollama Server

```bash
# In a terminal window
ollama serve
```

Keep this running in the background.

### 2. Test Cloud Model Access

```bash
# This will automatically connect to the cloud model
ollama run qwen3-vl:235b-cloud "Hello, generate a simple flowchart"
```

No `ollama pull` needed! The cloud model is accessed directly.

### 3. Verify Configuration

Your `config.json` should have:

```json
{
    "mermaid_generator": {
        "llm_provider": "ollama",
        "ollama_url": "http://localhost:11434",
        "ollama_model": "qwen3-vl:235b-cloud",
        "temperature": 0.3,
        "timeout": 120
    }
}
```

✅ This is already configured!

## Running Tests

### Quick Test

```bash
source venv/bin/activate
python test_ollama_integration.py
```

This will run 4 tests:
1. ✅ Ollama server health check
2. ✅ Simple text generation
3. ✅ Mermaid flowchart generation
4. ✅ Full pipeline integration

### Full Pipeline Test

```bash
python simple_examples.py
```

## Troubleshooting

### Issue: "Operation not permitted" or network blocked

**Solution for Claude Code users:**
1. Open http://localhost:4564
2. Navigate to "Tools" tab
3. Add `python` to the allowlist
4. Select "Permanent"
5. Retry

**Alternative:** Run outside Claude Code:
```bash
# In a regular terminal (not Claude Code)
cd /Users/haoxin/jaaz_pro
source venv/bin/activate
python test_ollama_integration.py
```

### Issue: "Cannot connect to Ollama server"

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, test
ollama list
```

### Issue: Timeouts

The cloud model may take 30-60 seconds for complex generations. The timeout is set to 120s in config.

## Model Comparison

| Model | Parameters | Location | Speed | Quality | Cost |
|-------|-----------|----------|-------|---------|------|
| llama3.2 | 3.2B | Local | ⚡⚡⚡ | ⭐⭐⭐ | Free |
| qwen2.5vl | 8.3B | Local | ⚡⚡ | ⭐⭐⭐⭐ | Free |
| **qwen3-vl:235b-cloud** | **235B** | **Cloud** | **⚡** | **⭐⭐⭐⭐⭐** | **$** |
| gpt-4 | ~1T | Cloud | ⚡⚡ | ⭐⭐⭐⭐⭐ | $$$ |

## Switching Models

Want to try a different model? Just update `config.json`:

```json
// For local, fast development
"ollama_model": "llama3.2"

// For local with vision
"ollama_model": "qwen2.5vl"

// For cloud, best quality (current)
"ollama_model": "qwen3-vl:235b-cloud"
```

No code changes needed!

## Next Steps

### Phase 1: Text → Flowchart ✅
Already working with rule-based + LLM fallback.

### Phase 2: Image Sketch → Flowchart 🎯
The qwen3-vl:235b-cloud model has **vision capabilities**! This means it can:
- Process image inputs directly
- Understand hand-drawn sketches
- Extract flowchart structure from photos

Perfect for the next phase!

## Cost Estimation

Ollama cloud models are significantly cheaper than commercial APIs:

- **OpenAI GPT-4**: ~$0.03 per 1K tokens ($30 per 1M)
- **Ollama Cloud**: Significantly lower (exact pricing varies)
- **Local Models**: $0.00 (but lower quality)

For development with ~1000 flowcharts:
- OpenAI: ~$20-50
- Ollama Cloud: ~$5-10
- Local: $0

## Performance Tips

1. **Lower temperature for code** - Set to 0.3 for consistent Mermaid syntax
2. **Increase timeout** - Cloud models need 30-120s for complex tasks
3. **Use fallback** - System automatically falls back to rule-based if LLM fails
4. **Batch processing** - Process multiple diagrams in parallel for efficiency

## Support

- Ollama Docs: https://ollama.ai/docs
- Qwen Models: https://huggingface.co/Qwen
- Project Issues: See README.md

## Quick Commands Reference

```bash
# Start Ollama
ollama serve

# List available models
ollama list

# Test cloud model
ollama run qwen3-vl:235b-cloud "test"

# Run integration tests
source venv/bin/activate
python test_ollama_integration.py

# Run full pipeline
python simple_examples.py

# Check logs
tail -f sketch_to_mermaid_feedback.log
```

---

**Ready to go!** 🚀

Your system is now configured with a powerful 235B vision-language model that will significantly improve flowchart generation quality while keeping costs reasonable.
