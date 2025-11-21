# Google Image Generation Research

## User Request Analysis

**User mentioned**: "use... nano banana for image generation"

### Clarification Needed

"Nano banana" is likely referring to one of the following:

1. **Google Imagen** - Google's text-to-image AI model
2. **Vertex AI Image Generation** - Google Cloud's image generation service
3. A typo or autocorrect error

---

## Google's Image Generation Options

### Option 1: Google Imagen (Most Likely)

**What is Imagen?**
- Google's state-of-the-art text-to-image diffusion model
- Competes with DALL-E, Midjourney, and Stable Diffusion
- Known for high photorealism and text understanding

**Access Methods:**

#### A) Vertex AI Imagen API (Google Cloud)
```python
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel

# Initialize
aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

# Generate image
model = ImageGenerationModel.from_pretrained("imagegeneration@005")
images = model.generate_images(
    prompt="professional flowchart diagram showing login process",
    number_of_images=1,
    guidance_scale=15,
    add_watermark=False
)

images[0].save("flowchart.png")
```

**Requirements:**
- Google Cloud Project with billing enabled
- Vertex AI API enabled
- Service account credentials
- Cost: ~$0.020 per image (512x512)

**Pros:**
- ✅ Official Google solution
- ✅ High quality results
- ✅ Well-maintained SDK
- ✅ Integrates with Google Cloud ecosystem

**Cons:**
- ⚠️ Requires Google Cloud account
- ⚠️ Pay-per-use (not free)
- ⚠️ More complex setup than open-source

---

#### B) Imagen 3 via Google AI Studio (Experimental)
- Available through Google AI Studio interface
- Currently limited API access
- May not be publicly available yet

---

### Option 2: Gemini with Image Generation (NOT Available)

**Status**: Gemini API **does NOT support image generation** (as of January 2025)

Gemini capabilities:
- ✅ Text generation
- ✅ Vision (image understanding)
- ❌ Image generation (NOT supported)

For image generation, you need a separate service like Imagen.

---

### Option 3: Open-Source Alternative (From Previous Research)

If "nano banana" was meant as a placeholder or if cost is a concern:

**Stable Diffusion XL Turbo** (Recommended Open-Source)
- Local, free, high-quality
- See `IMAGE_GENERATION_RESEARCH.md` for details

---

## Recommended Implementation Strategy

### Strategy A: Use Google Imagen (If that's what user meant)

**Setup:**

1. **Install dependencies:**
```bash
pip install google-cloud-aiplatform pillow
```

2. **Create Imagen client** (`whiteboard_pipeline/components/imagen_client.py`):
```python
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image
from typing import Optional
import logging

class ImagenClient:
    """Client for Google Imagen text-to-image generation"""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize Vertex AI
        project_id = config.get('google_cloud_project_id')
        location = config.get('google_cloud_location', 'us-central1')

        aiplatform.init(project=project_id, location=location)

        # Load model
        self.model = ImageGenerationModel.from_pretrained("imagegeneration@005")

        self.logger.info("Imagen client initialized")

    async def generate_diagram_image(self, description: str,
                                     style: str = "professional diagram") -> Optional[Image.Image]:
        """Generate diagram image from text description"""

        prompt = self._create_diagram_prompt(description, style)

        try:
            images = self.model.generate_images(
                prompt=prompt,
                number_of_images=1,
                guidance_scale=15,
                add_watermark=False
            )

            return images[0]._pil_image

        except Exception as e:
            self.logger.error(f"Imagen generation failed: {e}")
            return None

    def _create_diagram_prompt(self, description: str, style: str) -> str:
        """Create optimized prompt for diagram generation"""
        return f"""{style}, {description},
        clean lines, simple shapes, white background,
        technical illustration, vector art style,
        minimalist design, high contrast, clear labels"""
```

3. **Configuration** (add to `config.json`):
```json
{
    "image_generator": {
        "provider": "imagen",
        "google_cloud_project_id": "YOUR_PROJECT_ID",
        "google_cloud_location": "us-central1",
        "default_style": "professional diagram"
    }
}
```

4. **Set up authentication:**
```bash
# Set environment variable for service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

---

### Strategy B: Clarify with User

**Create placeholder implementation** and ask user to confirm which service they want:

1. Google Imagen (Vertex AI) - Cloud-based, paid
2. Stable Diffusion (open-source) - Local, free
3. Something else they had in mind?

---

## Cost Comparison

| Service | Cost per Image | Quality | Setup Complexity |
|---------|---------------|---------|------------------|
| **Google Imagen** | $0.02-0.04 | ⭐⭐⭐⭐⭐ | Medium |
| **SDXL-Turbo** | Free | ⭐⭐⭐⭐ | Medium |
| **OpenAI DALL-E 3** | $0.04-0.08 | ⭐⭐⭐⭐⭐ | Easy |

---

## Implementation Steps (Assuming Imagen)

### Phase 1: Basic Setup
- [ ] Clarify with user if "nano banana" = Imagen
- [ ] Set up Google Cloud project
- [ ] Enable Vertex AI API
- [ ] Create service account credentials
- [ ] Install google-cloud-aiplatform

### Phase 2: Create ImagenClient
- [ ] Create `imagen_client.py` component
- [ ] Implement basic text-to-image generation
- [ ] Add diagram-specific prompt engineering
- [ ] Test with sample prompts

### Phase 3: Integration
- [ ] Integrate with pipeline
- [ ] Add to demo.py examples
- [ ] Update documentation
- [ ] Test end-to-end workflow

---

## Questions for User

1. **Confirmation**: Did you mean "Imagen" when you said "nano banana"?
2. **Google Cloud**: Do you have a Google Cloud project set up?
3. **Budget**: Are you comfortable with pay-per-use pricing (~$0.02/image)?
4. **Alternative**: If cost is a concern, should we use open-source Stable Diffusion instead?

---

## Next Steps

**RECOMMENDATION**: Create placeholder `ImagenClient` with API key fields, then confirm with user which service they actually want before completing implementation.

This allows flexibility to switch between:
- Google Imagen (if that's what they meant)
- Stable Diffusion (if they want free/local)
- Other services they might clarify

---

## File to Create

`whiteboard_pipeline/components/imagen_client.py` - Ready to implement once confirmed!
