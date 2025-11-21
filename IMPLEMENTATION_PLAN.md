# Phase 2 & 3 Implementation Plan

## Overview

Building two high-priority visual features in parallel:
- **Phase 2**: Sketch → Flowchart (Understanding)
- **Phase 3**: Text → Image (Generation)

---

## Phase 2: Sketch → Flowchart Implementation

### Architecture

```
Whiteboard Photo/Sketch (PNG/JPG/PDF)
    ↓
Image Preprocessing
├─ Rotation correction
├─ Noise reduction
├─ Contrast enhancement
└─ Resize/normalize
    ↓
Vision-Language Model (qwen2.5vl)
├─ Understand diagram structure
├─ Identify shapes and connections
└─ Extract text from elements
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

**2. Vision Analyzer**
```python
class VisionAnalyzer:
    def __init__(self, model="qwen2.5vl"):
        self.vlm = load_vision_model(model)

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

### Tasks

- [ ] Create `ImageInputHandler` component
- [ ] Implement image preprocessing pipeline
- [ ] Integrate qwen2.5vl for visual understanding
- [ ] Create prompt templates for diagram analysis
- [ ] Parse VLM output into structured format
- [ ] Update pipeline to handle IMAGE input type
- [ ] Test with sample whiteboard photos
- [ ] Benchmark accuracy and performance

---

## Phase 3: Text → Image Generation

### Architecture

```
Text Description
    ↓
Diagram Prompt Builder
├─ Add diagram-specific keywords
├─ Specify visual style
└─ Add negative prompts
    ↓
Stable Diffusion XL Turbo
├─ Local inference (GPU/MPS)
├─ 4-8 generation steps
└─ Fast turbo variant
    ↓
Post-Processing
├─ Resize if needed
├─ Format conversion
└─ Quality validation
    ↓
Generated Image (PNG)
```

### Components

**1. Image Generator**
```python
from diffusers import DiffusionPipeline
import torch

class DiagramImageGenerator:
    def __init__(self, model="sdxl-turbo"):
        self.pipe = self._load_model(model)

    def _load_model(self, model_name):
        pipe = DiffusionPipeline.from_pretrained(
            f"stabilityai/{model_name}",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        # Use GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif torch.backends.mps.is_available():
            pipe = pipe.to("mps")
        return pipe

    async def generate(self, description: str) -> Image:
        prompt = self._build_diagram_prompt(description)
        negative = self._get_negative_prompt()

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=4,
            guidance_scale=0.0
        ).images[0]

        return image

    def _build_diagram_prompt(self, description: str) -> str:
        return f"""professional flowchart diagram, {description},
        clean lines, simple shapes, white background,
        technical illustration, vector art style,
        minimalist design, high contrast, clear labels"""

    def _get_negative_prompt(self) -> str:
        return """photo, photograph, realistic, 3d render,
        cluttered, messy, handwriting, blurry, low quality"""
```

**2. Integration**
```python
# Add to pipeline
class VisualPipeline:
    def __init__(self, config):
        self.text_to_mermaid = TextToMermaid(config)
        self.sketch_to_mermaid = SketchToMermaid(config)
        self.image_generator = DiagramImageGenerator(config)

    async def process(self, input_data):
        if input_data.type == "text" and input_data.want_image:
            # Generate Mermaid
            mermaid = await self.text_to_mermaid.generate(input_data)

            # Generate visual image
            image = await self.image_generator.generate(input_data.content)

            return {"mermaid": mermaid, "image": image}
```

### Tasks

- [ ] Install diffusers and dependencies
- [ ] Create `DiagramImageGenerator` component
- [ ] Test basic image generation
- [ ] Create diagram-specific prompt templates
- [ ] Implement post-processing pipeline
- [ ] Add GPU/CPU fallback handling
- [ ] Integrate with main pipeline
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
