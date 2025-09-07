# Whiteboard Processing Pipeline - Sketch to Mermaid

A production-ready Python pipeline that converts whiteboard sketches and text into Mermaid flowcharts. **Start small philosophy** - focused on getting the core "Sketch → Mermaid" workflow working perfectly first.

## 🎯 Core Workflow

```
[ Whiteboard Input: Sketch / Text ]
          │
          ▼
   ┌──────────────┐
   │ Mandatory    │ ← PaddleOCR + EasyOCR backup
   │ OCR Parser   │   (No fallback to text-only)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  vLLM Engine │ ← Production vLLM (not hand-rolled)
   │              │   Qwen-VL → CREATE_FLOWCHART intent
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Mermaid    │ ← GPT-4/Claude → .mmd files
   │  Generator   │   With intelligent fallbacks
   └──────┬───────┘
          │
          ▼
[ 📄 Mermaid Flowchart + 📊 Comprehensive Feedback Logs ]
```

## 🚀 Key Improvements (Production Ready)

✅ **Use vLLM** — Production-ready model serving, not hand-rolled HTTP calls  
✅ **OCR is mandatory** — Reliable text extraction with PaddleOCR + EasyOCR backup  
✅ **Start small** — Perfect the core "Sketch → Mermaid" workflow first  
✅ **Log everything** — Comprehensive feedback collection for fine-tuning  
✅ **pytest** — Professional testing framework instead of unittest  

## 📦 Quick Start

### Installation

```bash
git clone <repository-url>
cd jaaz_pro

# Install dependencies (now streamlined)
pip install -r requirements.txt
```

### Basic Usage - Core Workflow

```python
import asyncio
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType

async def main():
    # Initialize the focused pipeline
    pipeline = SimpleSketchToMermaidPipeline()
    
    # Input: Text describing a process
    input_data = WhiteboardInput(
        content="User login: 1. Enter credentials 2. Validate 3. Grant access",
        input_type=InputType.TEXT
    )
    
    # Process: Sketch → Mermaid
    result = await pipeline.process_sketch_to_mermaid(input_data)
    
    if result.success:
        print(f"✅ Generated Mermaid flowchart: {result.outputs[0].file_path}")
        print(result.outputs[0].content)
    else:
        print(f"❌ Error: {result.error_message}")

asyncio.run(main())
```

### Configuration (Simplified)

```json
{
    "pipeline": {
        "log_level": "INFO",
        "log_file": "sketch_to_mermaid_feedback.log"
    },
    "input_parser": {
        "ocr_confidence_threshold": 0.3,
        "mandatory_ocr": true
    },
    "vlm_engine": {
        "model_name": "Qwen/Qwen-VL-Chat",
        "fallback_enabled": true
    },
    "mermaid_generator": {
        "llm_provider": "openai",
        "api_key": "${OPENAI_API_KEY}"
    }
}
```

## 🏗️ Architecture - Focused & Production-Ready

### Core Components (Simplified)
- **SimpleSketchToMermaidPipeline**: Main orchestrator focused on the core workflow
- **InputParser**: Mandatory OCR with PaddleOCR + EasyOCR backup 
- **VLMEngine**: Production vLLM integration (not hand-rolled API calls)
- **MermaidFlowGenerator**: Enhanced with comprehensive logging and fallbacks

### Production Features
- **vLLM Integration**: Use `AsyncLLMEngine` directly, not HTTP endpoints
- **Mandatory OCR**: Never skip text extraction, multiple engine fallbacks
- **Comprehensive Logging**: Every step logged for feedback and fine-tuning
- **Smart Fallbacks**: Continue working even when external services fail
- **pytest Suite**: Professional testing with async support and mocking

## 📊 Feedback Collection (Log Everything)

The pipeline logs every step for future fine-tuning:

```python
# Detailed session logging
session_log = {
    'session_id': 'session_20240101_120000',
    'total_duration': 2.5,
    'success': True,
    'steps': [
        {'step': 'input_parsing', 'duration': 0.5, 'success': True},
        {'step': 'intent_extraction', 'duration': 1.0, 'success': True}, 
        {'step': 'mermaid_generation', 'duration': 1.0, 'success': True}
    ]
}

# Performance analytics
analytics = pipeline.get_session_analytics()
# Returns: success_rate, avg_duration, step_performance, etc.
```

## 🧪 Testing (Professional pytest)

```bash
# Run the focused test suite
pytest test_simple_pipeline.py -v

# With coverage
pytest test_simple_pipeline.py --cov=whiteboard_pipeline

# Run specific test categories
pytest test_simple_pipeline.py::TestSimpleSketchToMermaidPipeline -v
```

## 📚 Examples

See `simple_examples.py` for comprehensive demonstrations:

- **Core Workflow**: Text → Mermaid conversion
- **Simulated Sketches**: OCR output → Mermaid
- **Iterative Processing**: Multiple workflows for feedback collection
- **Health Monitoring**: System status and component checks
- **Error Handling**: Graceful degradation and recovery

```bash
python simple_examples.py
```

## ⚙️ Production Setup

### 1. vLLM Server (Production Model Serving)

```bash
# Start vLLM server with Qwen-VL
pip install vllm
vllm serve Qwen/Qwen-VL-Chat --port 8000 --gpu-memory-utilization 0.9
```

### 2. Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
# Optional: ANTHROPIC_API_KEY for Claude fallback
```

### 3. Dependencies (Streamlined)

```bash
# Core + vLLM
pip install vllm>=0.2.0

# Mandatory OCR
pip install paddleocr>=2.6.0 easyocr>=1.6.0

# Testing
pip install pytest>=7.0.0 pytest-asyncio>=0.21.0
```

## 🔍 Monitoring & Health Checks

```python
# Built-in health monitoring
health = await pipeline.health_check()
print(f"Pipeline status: {health['pipeline']}")
print(f"Components: {health['components']}")

# Performance analytics
analytics = pipeline.get_session_analytics()
print(f"Success rate: {analytics['success_rate']:.1%}")
print(f"Average processing time: {analytics['average_duration']:.2f}s")
```

## 🚧 Roadmap (After Core is Perfect)

1. **Phase 1** ✅: Perfect Sketch → Mermaid (Current)
2. **Phase 2**: Add Image → Mermaid (actual sketch processing)
3. **Phase 3**: Add other output formats (reports, diagrams)
4. **Phase 4**: Fine-tune with collected feedback data

## 🔧 Key Technical Decisions

- **vLLM over HTTP**: Direct `AsyncLLMEngine` integration for production performance
- **Mandatory OCR**: Never compromise on text extraction quality
- **Start Small**: Perfect one workflow before expanding
- **Log Everything**: Every operation captured for continuous improvement
- **pytest**: Professional testing framework with async support

## 📈 Why This Approach Works

1. **Production Ready**: Uses proper production tools (vLLM, not hand-rolled)
2. **Reliable**: Mandatory OCR with multiple fallbacks ensures robustness
3. **Focused**: Master one workflow completely before expansion
4. **Data-Driven**: Comprehensive logging enables continuous improvement
5. **Testable**: pytest suite ensures quality and regression prevention

## 🤝 Contributing

1. Focus on the core Sketch → Mermaid workflow
2. Add comprehensive logging to any new features
3. Write pytest tests for all new functionality
4. Use production-ready dependencies (vLLM, not hand-rolled APIs)
5. Maintain the "start small" philosophy

## 📄 License

[Add your license information here]