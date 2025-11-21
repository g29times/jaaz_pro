# Phase 2 & 3 Implementation Plan

## 🚨 UPDATED ARCHITECTURE: Google AI Stack

**Date**: 2025-01-20
**Status**: In Progress

### New Technology Stack
- **Primary LLM**: Google Gemini (gemini-1.5-flash / gemini-1.5-pro)
- **Fallback LLM**: Ollama (qwen2.5vl:latest) - local
- **Image Generation**: Google Imagen (via Vertex AI)
- **Vision Understanding**: Gemini Vision API

### Architecture Decision
- Use Google's AI ecosystem as PRIMARY
- Keep Ollama as FALLBACK for offline/development
- Centralized configuration with API key placeholders

---

## Overview

Building two high-priority visual features in parallel:
- **Phase 2**: Sketch → Flowchart (Understanding) - Using Gemini Vision
- **Phase 3**: Text → Image (Generation) - Using Google Imagen

---

## Phase 2: Sketch → Flowchart Implementation

### Updated Architecture

```
Whiteboard Photo/Sketch (PNG/JPG/PDF)
    ↓
Image Preprocessing
├─ Rotation correction
├─ Noise reduction
├─ Contrast enhancement
└─ Resize/normalize
    ↓
Google Gemini Vision API ⭐ (PRIMARY)
├─ Understand diagram structure
├─ Identify shapes and connections
└─ Extract text from elements
    ↓ (fallback)
qwen2.5vl via Ollama (LOCAL FALLBACK)
    ↓
Mermaid Code Generation
    ↓
Clean Flowchart Diagram
```

### Components

**1. Image Input Handler**
```python
class ImageInputHandler:
    async def load_image(self, path: str) -> Image
    async def preprocess(self, image: Image) -> Image
    def detect_rotation(self, image: Image) -> float
    def enhance_contrast(self, image: Image) -> Image
```

**2. Vision Analyzer (Updated)**
```python
class VisionAnalyzer:
    def __init__(self):
        self.gemini_client = GeminiClient(config)  # PRIMARY
        self.ollama_client = OllamaClient(config)  # FALLBACK

    async def analyze_sketch(self, image: Image) -> SketchAnalysis:
        # Try Gemini Vision first
        try:
            return await self.gemini_client.analyze_image(image)
        except:
            # Fallback to local qwen2.5vl
            return await self.ollama_client.analyze_image(image)
```

    async def analyze_sketch(self, image: Image) -> SketchAnalysis:
        # Use VLM to understand the sketch
        pass

    def extract_elements(self, analysis) -> List[DiagramElement]:
        # Parse VLM output into structured elements
        pass
```

**3. Updated Pipeline**
```python
# Extend existing pipeline
if input_type == InputType.IMAGE:
    # Load and preprocess image
    image = await image_handler.load(input_path)
    image = await image_handler.preprocess(image)

    # Analyze with vision model
    analysis = await vision_analyzer.analyze_sketch(image)
    elements = vision_analyzer.extract_elements(analysis)

    # Generate Mermaid from elements
    mermaid = await mermaid_generator.generate_from_elements(elements)
```

### Tasks (Updated)

- [ ] Create `ImageInputHandler` component
- [ ] Implement image preprocessing pipeline
- [x] Integrate Gemini Vision for visual understanding (PRIMARY)
- [ ] Keep qwen2.5vl as fallback for offline use
- [ ] Create prompt templates for diagram analysis
- [ ] Parse VLM output into structured format
- [ ] Update pipeline to handle IMAGE input type
- [ ] Test with sample whiteboard photos
- [ ] Benchmark accuracy and performance

---

## Phase 3: Text → Image Generation (UPDATED)

### Updated Architecture (Google Imagen)

```
Text Description
    ↓
Diagram Prompt Builder
├─ Add diagram-specific keywords
├─ Specify visual style
└─ Optimize for Imagen model
    ↓
Google Imagen (Vertex AI) ⭐ (PRIMARY)
├─ Cloud-based inference
├─ High-quality image generation
└─ ~$0.02 per image
    ↓
Post-Processing
├─ Resize if needed
├─ Format conversion
└─ Quality validation
    ↓
Generated Image (PNG)
```

**Key Change**: Using Google Imagen instead of local Stable Diffusion
- **Pros**: Better quality, no GPU requirements, official Google solution
- **Cons**: Requires Google Cloud setup, pay-per-use

### Components (Updated)

**1. Image Generator (ImagenClient)**
```python
from whiteboard_pipeline.components.imagen_client import ImagenClient

class DiagramImageGenerator:
    def __init__(self, config):
        self.imagen_client = ImagenClient(config)

    async def generate(self, description: str) -> Image:
        """Generate diagram image using Google Imagen"""
        image = await self.imagen_client.generate_diagram_image(
            description=description,
            style="professional diagram"
        )
        return image

    async def generate_from_mermaid(self, mermaid_code: str) -> Image:
        """Generate visual image from Mermaid code"""
        return await self.imagen_client.generate_from_mermaid(mermaid_code)

```

**2. Integration (Updated)**
```python
# Add to pipeline
class VisualPipeline:
    def __init__(self, config):
        self.gemini_client = GeminiClient(config)  # Text → Mermaid
        self.imagen_client = ImagenClient(config)  # Text → Image

    async def process(self, input_data):
        if input_data.type == "text" and input_data.want_image:
            # Generate Mermaid flowchart code
            mermaid = await self.gemini_client.generate_mermaid_from_text(
                input_data.content
            )

            # Generate visual image from the same description
            image = await self.imagen_client.generate_diagram_image(
                input_data.content
            )

            return {"mermaid": mermaid, "image": image}
```

### Tasks (Updated)

- [x] Research Google Imagen API
- [x] Create `ImagenClient` component
- [ ] Set up Google Cloud project and credentials
- [ ] Install google-cloud-aiplatform dependencies
- [ ] Test basic image generation with Imagen
- [ ] Create diagram-specific prompt templates for Imagen
- [ ] Integrate with main pipeline
- [ ] Test end-to-end text → image workflow
- [ ] Test with real diagram descriptions
- [ ] Benchmark generation speed and quality

---

## Dependencies

### Phase 2 (Sketch Understanding)
```bash
# Already have:
- qwen2.5vl model via Ollama ✅
- PIL/Pillow for image handling ✅
- OpenCV for preprocessing ✅

# May need:
pip install pillow opencv-python numpy
```

### Phase 3 (Image Generation)
```bash
# New dependencies:
pip install diffusers transformers accelerate
pip install torch torchvision  # Or MPS version for Mac

# Model download (automatic on first run):
# SDXL-Turbo: ~6.9GB
```

---

## Development Order

### Week 1: Phase 3 (Easier to Start)
1. Install diffusers
2. Create ImageGenerator component
3. Test basic generation
4. Create prompt templates
5. Integration demo

### Week 2: Phase 2 (More Complex)
1. Create ImageInputHandler
2. Implement preprocessing
3. Integrate qwen2.5vl vision
4. Parse VLM outputs
5. Full pipeline integration

### Week 3: Polish & Testing
1. Test both features together
2. Performance optimization
3. Error handling
4. Documentation
5. Demo examples

---

## Testing Strategy

### Phase 2 Tests
- [ ] Load various image formats (PNG, JPG, PDF)
- [ ] Handle rotated images
- [ ] Process noisy/low-quality photos
- [ ] Extract text from diagrams
- [ ] Recognize different flowchart shapes
- [ ] Generate valid Mermaid code
- [ ] Benchmark accuracy vs hand-drawn samples

### Phase 3 Tests
- [ ] Generate from simple descriptions
- [ ] Generate from complex descriptions
- [ ] Test different diagram types
- [ ] Verify image quality
- [ ] Test on CPU vs GPU
- [ ] Measure generation speed
- [ ] Validate output format

---

## Success Metrics

### Phase 2
- ✅ 90%+ accuracy on clean whiteboard photos
- ✅ < 5s processing time per image
- ✅ Handles rotated/skewed images
- ✅ Extracts text correctly
- ✅ Identifies shapes and connections

### Phase 3
- ✅ Generates high-quality diagram images
- ✅ < 10s generation time (GPU)
- ✅ Professional, clean appearance
- ✅ Follows text description accurately
- ✅ Multiple diagram styles supported

---

## Next Steps

1. Start with Phase 3 (image generation) - easier to implement
2. Install diffusers and test basic generation
3. Create DiagramImageGenerator component
4. Then move to Phase 2 (sketch understanding)
5. Parallel development where possible

Ready to begin implementation! 🚀
