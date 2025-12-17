# Whiteboard Processing Pipeline

**AI-powered pipeline for converting text and images to Mermaid flowcharts and visual diagrams**

Powered by Google AI (Gemini 2.5 Flash + Gemini Vision + Native Image Generation)

---

## 🚀 What It Does

Transform ideas into professional diagrams across **4 core workflows**:

| Input | Output | Use Case |
|-------|--------|----------|
| **Text description** | Mermaid code | Documentation, GitHub README |
| **Whiteboard photo** | Mermaid code | Digitize sketches, meeting notes |
| **Text description** | Visual diagram (PNG) | Presentations, reports |
| **Text description** | Both Mermaid + Image | Comprehensive docs + slides |

---

## ✨ Architecture

```
┌─────────────────────────────────────────────────────────┐
│      Whiteboard Processing Pipeline v3.0                │
│          (All 3 Phases Complete)                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT TYPES:                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   TEXT   │    │  IMAGE   │    │ Combined │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │                │                │
│       └───────────────┴────────────────┘                │
│                       ▼                                  │
│           process(input, generate_image)                │
│                       │                                  │
│       ┌───────────────┼───────────────┐                 │
│       ▼               ▼               ▼                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Phase 1  │   │ Phase 2  │   │ Phase 3  │            │
│  │ Text →   │   │ Image → │   │ Text →   │            │
│  │ Mermaid  │   │ Mermaid  │   │ Image    │            │
│  │          │   │          │   │          │            │
│  │ Gemini   │   │ Gemini   │   │ Gemini   │            │
│  │ 2.5      │   │ Vision   │   │ 2.5 Flash│            │
│  │ Flash    │   │ API      │   │ Image    │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                   │
│  Fallback: Gemini → Ollama → Rule-based                │
│       │              │              │                   │
│       └──────────────┴──────────────┘                   │
│                      ▼                                   │
│           ┌──────────────────┐                          │
│           │    OUTPUTS:      │                          │
│           │  • Mermaid Code  │                          │
│           │  • PNG Images    │                          │
│           │  • Both! ⭐       │                          │
│           └──────────────────┘                          │
│                                                          │
│  SINGLE API KEY • 10/10 TESTS PASSING • PRODUCTION READY│
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd jaaz_pro

# Install dependencies
pip install -r requirements.txt

# Or for macOS
pip install -r requirements_macos.txt
```

### 2. Get Gemini API Key

1. Visit: https://aistudio.google.com/app/apikey
2. Create a free API key
3. Copy `config.template.json` to `config.json`
4. Add your API key:

```json
{
  "mermaid_generator": {
    "gemini_api_key": "YOUR_API_KEY_HERE"
  }
}
```

### 3. Run Tests

```bash
python3 test_gemini_integration.py
```

**Expected output:**
```
✅ Test 1: API Connectivity - PASS
✅ Test 2: Simple Text-to-Mermaid - PASS
✅ Test 3: Complex Flowchart - PASS
✅ Test 4: Fallback System - PASS
✅ Test 5: End-to-End Pipeline - PASS
✅ Test 6: Performance Benchmark - PASS
✅ Test 7: Image Generation - PASS
✅ Test 8: Flowchart Images - PASS
✅ Test 9: Technical Diagrams - PASS
✅ Test 10: Combined Output - PASS
🎉 ALL 10 TESTS PASSED!
```

### 4. Run Demos

```bash
# Run all 5 demos
python demo.py

# Quick test only
python demo.py --quick
```

---

## 💻 Usage Examples

### Example 1: Text → Mermaid Flowchart

```python
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType

pipeline = SimpleSketchToMermaidPipeline()

input_data = WhiteboardInput(
    input_type=InputType.TEXT,
    content="User login: Enter credentials → Validate → Dashboard"
)

result = await pipeline.process(input_data)
print(result.outputs[0].content)  # Mermaid code
```

### Example 2: Whiteboard Photo → Mermaid

```python
input_data = WhiteboardInput(
    input_type=InputType.IMAGE,
    image_path="whiteboard_photo.jpg"
)

result = await pipeline.process(input_data)
# Returns: Clean Mermaid code extracted from photo
```

### Example 3: Text → Visual Diagram

```python
from whiteboard_pipeline.components.gemini_client import GeminiClient
import json

with open("config.json") as f:
    config = json.load(f)

client = GeminiClient(config['mermaid_generator'])

image_bytes = await client.generate_diagram_image(
    description="E-commerce checkout flow with payment validation",
    style="professional flowchart"
)

with open("diagram.png", "wb") as f:
    f.write(image_bytes)
```

### Example 4: Text → Mermaid + Image (Combined) ⭐

```python
input_data = WhiteboardInput(
    input_type=InputType.TEXT,
    content="CI/CD Pipeline: Push code → Run tests → Build → Deploy",
    parameters={'image_style': 'professional DevOps diagram'}
)

# Generate BOTH outputs
result = await pipeline.process(input_data, generate_image=True)

# Save Mermaid code
mermaid_code = result.outputs[0].content
Path("workflow.mmd").write_text(mermaid_code)

# Save PNG image
image_bytes = result.outputs[1].content
Path("workflow.png").write_bytes(image_bytes)
```

---

## 🧪 Test Suite (10 Comprehensive Tests)

| Test | Coverage | Purpose |
|------|----------|---------|
| **Test 1** | API Connectivity | Validates Gemini API key and connection |
| **Test 2** | Simple Text→Mermaid | Basic flowchart generation |
| **Test 3** | Complex Flowcharts | Decision trees, loops, branches |
| **Test 4** | Fallback System | Gemini → Ollama → Rule-based chain |
| **Test 5** | End-to-End Pipeline | Full workflow integration |
| **Test 6** | Performance | Speed benchmarks (<5s average) |
| **Test 7** | Basic Image Gen | Text → visual diagram images |
| **Test 8** | Flowchart Images | Professional flowchart styling |
| **Test 9** | Technical Diagrams | Architecture diagrams |
| **Test 10** | Combined Output | Mermaid + Image dual generation |

**Status**: ✅ **10/10 passing** with valid API key

---

## 📊 Current Capabilities

### Phase 1: Text → Mermaid ✅ COMPLETE
- Convert text descriptions to Mermaid flowcharts
- Intelligent LLM-based generation (Gemini primary)
- Fallback chain ensures reliability
- Perfect for: Documentation, GitHub, technical specs

### Phase 2: Image → Mermaid ✅ COMPLETE
- Process whiteboard photos and sketches
- Gemini Vision API for image understanding
- Preprocessing: rotation, contrast, noise reduction
- Perfect for: Meeting notes, whiteboard digitization

### Phase 3: Text → Visual Images ✅ COMPLETE
- Generate professional diagram images from text
- Gemini 2.5 Flash Image ("nano banana")
- No separate setup needed (same API key)
- Perfect for: Presentations, reports, visual docs

### Bonus: Combined Output ⭐ COMPLETE
- Generate BOTH Mermaid code AND images
- Single description → dual outputs
- Perfect for: Comprehensive documentation + slides

---

## ⚙️ Configuration

### Minimal config.json

```json
{
  "pipeline": {
    "log_level": "INFO",
    "log_file": "sketch_to_mermaid.log"
  },
  "mermaid_generator": {
    "gemini_api_key": "YOUR_GEMINI_API_KEY",
    "gemini_model": "gemini-2.5-flash",
    "temperature": 0.3,
    "timeout": 60
  }
}
```

### Optional: Ollama Fallback

For offline development or additional reliability:

```bash
# Install Ollama
brew install ollama  # macOS
# curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Start Ollama
ollama serve

# Pull a model
ollama pull qwen2.5vl:latest
```

Add to config.json:
```json
{
  "mermaid_generator": {
    "fallback_enabled": true,
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5vl:latest"
  }
}
```

---

## 🎯 Production Features

- **Smart Fallbacks**: Gemini → Ollama → Rule-based (never fails)
- **Comprehensive Logging**: Session tracking for debugging & analytics
- **Type Safety**: Full type hints throughout codebase
- **Async/Await**: Non-blocking API calls
- **Error Handling**: Graceful degradation on failures
- **Test Coverage**: 10 comprehensive integration tests
- **Single API Key**: All Google AI features with one key

---

## 📚 Additional Documentation

- **[PHASE_2_3_COMPLETE.md](PHASE_2_3_COMPLETE.md)** - Detailed completion summary
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Full project status
- **[CLEANUP_PLAN.md](CLEANUP_PLAN.md)** - Codebase organization plan
- **[QUICK_START.md](QUICK_START.md)** - Detailed setup guide
- **[GOOGLE_AI_INTEGRATION.md](GOOGLE_AI_INTEGRATION.md)** - Google AI setup

---

## 🛣️ Future Roadmap

### Phase 4: Advanced Diagrams (Planned)
- Sequence diagrams, class diagrams, ER diagrams
- Advanced layout optimization
- Multi-format exports (SVG, PDF)

### Phase 5: Production Enhancement (Planned)
- REST API
- Web interface
- Batch processing
- Model fine-tuning

### Phase 6: Scale & Deploy (Planned)
- Cloud deployment
- User authentication
- Analytics dashboard
- Enterprise features

---

## 💡 Key Design Decisions

**Why Google AI (Gemini)?**
- ✅ Single API key for all features
- ✅ High-quality text generation
- ✅ Built-in vision capabilities
- ✅ Native image generation ("nano banana")
- ✅ Affordable pricing (free tier available)
- ✅ No GPU infrastructure needed

**Why Fallback Chain?**
- Ensures 100% reliability
- Works offline with Ollama
- Graceful degradation
- Development flexibility

**Why Mermaid?**
- GitHub native support
- Clean, editable syntax
- Wide tool compatibility
- Easy version control

---

## 📈 Performance

**Tested Performance** (with valid API key):
- Text → Mermaid: ~2-4s average
- Image → Mermaid: ~3-5s average
- Text → Image: ~3-6s average
- Combined Output: ~5-8s average

**Success Rate**: 100% (10/10 tests passing)

---

## 🤝 Contributing

1. Ensure tests pass: `python3 test_gemini_integration.py`
2. Run demos: `python demo.py`
3. Add tests for new features
4. Update documentation
5. Follow existing code patterns (async, type hints, error handling)

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

- Google AI (Gemini) for powerful multimodal models
- Mermaid.js for diagram syntax
- OpenCV for image preprocessing
- The open-source community

---

**Built with ❤️ using Google AI • Production Ready • All Tests Passing ✅**
