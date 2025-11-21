# Image Generation Models - Research & Implementation Plan

## Available Open-Source Image Generation Models

### Option 1: Stable Diffusion via Diffusers (Recommended)
**Best for**: High quality, flexibility, local control

**Models:**
- **SDXL (Stable Diffusion XL)**: 1024x1024, best quality
- **SD 1.5**: 512x512, faster, lower VRAM
- **LCM (Latent Consistency Models)**: Fast generation (4-8 steps)
- **SDXL-Turbo**: Fast SDXL variant

**Setup:**
```bash
pip install diffusers transformers accelerate torch
```

**Pros:**
- ✅ Best quality for diagram generation
- ✅ Full control over parameters
- ✅ Large model ecosystem
- ✅ Active community support
- ✅ Works offline after download

**Cons:**
- ⚠️ Requires GPU (6GB+ VRAM for SDXL)
- ⚠️ Large model downloads (3-7GB)

---

### Option 2: Flux via Diffusers
**Best for**: State-of-the-art quality

**Models:**
- **Flux.1-schnell**: Fast variant, Apache 2.0 license
- **Flux.1-dev**: Higher quality, non-commercial use

**Requirements:**
- GPU with 12GB+ VRAM for dev
- GPU with 6GB+ VRAM for schnell

**Pros:**
- ✅ Cutting-edge quality
- ✅ Better prompt following
- ✅ Apache 2.0 license (schnell)

**Cons:**
- ⚠️ Higher VRAM requirements
- ⚠️ Larger model size

---

### Option 3: Ollama with LLaVA (Not Recommended for Generation)
**Status**: Ollama focuses on text/vision understanding, NOT image generation

Ollama currently doesn't support image generation models. It's designed for:
- Text generation (LLMs)
- Vision understanding (VLMs like qwen2.5vl)

---

### Option 4: ComfyUI (Advanced Users)
**Best for**: Complex workflows, fine control

**Features:**
- Node-based workflow editor
- Multiple model support
- ControlNet, LoRA support
- Advanced post-processing

**Pros:**
- ✅ Very powerful
- ✅ Visual workflow building
- ✅ Community workflows

**Cons:**
- ⚠️ Complex setup
- ⚠️ Steeper learning curve

---

## Recommended Solution: Diffusers Library

### Why Diffusers?
1. **Python-native**: Easy integration with our pipeline
2. **Flexible**: Multiple models (SDXL, SD 1.5, LCM, Flux)
3. **Production-ready**: Used by HuggingFace
4. **Local**: Works offline, no API costs
5. **Well-documented**: Extensive guides and examples

### Implementation Plan

#### Phase 3.1: Basic Image Generation
```python
from diffusers import DiffusionPipeline
import torch

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")  # or "mps" for Mac M1/M2

# Generate
image = pipe(
    prompt="flowchart diagram showing login process",
    num_inference_steps=4,  # Fast with turbo
    guidance_scale=0.0
).images[0]

image.save("flowchart.png")
```

#### Phase 3.2: Diagram-Specific Prompts
Create templates for different diagram types:
```python
FLOWCHART_PROMPT = """
professional flowchart diagram, {description},
clean lines, simple shapes, white background,
technical illustration style, vector art look,
minimalist design, high contrast
"""

DIAGRAM_NEGATIVE = """
photo, photograph, realistic, 3d render,
cluttered, messy, handwriting, sketchy
"""
```

#### Phase 3.3: Integration with Pipeline
```python
class ImageGenerator:
    def __init__(self, model="sdxl-turbo"):
        self.pipe = self._load_model(model)

    async def generate_flowchart_image(self, description: str) -> Path:
        prompt = self._create_diagram_prompt(description)
        image = self.pipe(prompt).images[0]
        return self._save_image(image)
```

---

## Model Comparison

| Model | Size | Speed | Quality | VRAM | License | Best For |
|-------|------|-------|---------|------|---------|----------|
| **SDXL-Turbo** | 6.9GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | 6GB | Free | **Recommended** |
| SD 1.5 | 3.4GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | 4GB | Free | Fast prototyping |
| LCM-SDXL | 6.9GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 6GB | Free | Fastest |
| Flux-schnell | 12GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 12GB | Apache 2.0 | Best quality |
| SDXL | 6.9GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | Free | High quality |

---

## CPU vs GPU

### GPU (Recommended)
- **NVIDIA**: CUDA support, best performance
- **Mac M1/M2/M3**: MPS (Metal Performance Shaders)
- **AMD**: ROCm support (experimental)

### CPU (Fallback)
- Possible but VERY slow (20-60x slower)
- Only practical for testing
- Use smallest models (SD 1.5)

---

## Installation Requirements

### Minimal Setup (SDXL-Turbo)
```bash
# Install dependencies
pip install diffusers transformers accelerate torch torchvision

# For Mac M1/M2/M3
pip install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

---

## Next Steps

1. ✅ Choose model: **SDXL-Turbo** (best balance)
2. ✅ Install diffusers library
3. ✅ Create `ImageGenerator` component
4. ✅ Test basic generation
5. ✅ Create diagram-specific prompts
6. ✅ Integrate with pipeline
7. ✅ Add post-processing (resize, format conversion)
8. ✅ Test with real use cases

---

## Example Use Cases

### Use Case 1: Generate Flowchart Image
```
Input: "User login process with authentication"
Output: Professional flowchart PNG image
```

### Use Case 2: Illustrate Process
```
Input: "E-commerce order fulfillment workflow"
Output: Illustrated diagram with icons and flow
```

### Use Case 3: Create Presentation Visuals
```
Input: "System architecture with microservices"
Output: Clean architectural diagram
```

---

## Fallback Strategy

If GPU not available:
1. Use smaller models (SD 1.5)
2. Reduce image size (512x512)
3. Fewer inference steps
4. Or: Provide external API option (Replicate, HuggingFace Inference API)

---

## Conclusion

**Recommended Stack:**
- **Primary**: SDXL-Turbo via Diffusers
- **Fast variant**: LCM-SDXL for speed
- **Best quality**: Flux-schnell (if GPU allows)
- **Fallback**: SD 1.5 for limited hardware

This gives us local, cost-free, high-quality image generation! 🎨
