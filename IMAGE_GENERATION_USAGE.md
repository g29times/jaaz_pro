# Image Generation Usage Guide

**Status**: Component Ready | Not Integrated | Requires Google Cloud Setup

---

## 🎯 Quick Answer

**Yes, image generation is available but:**
- ✅ Component exists and is fully functional
- ❌ NOT integrated into the main pipeline (manual use only)
- ⚠️ Requires Google Cloud setup (not free)

---

## 🚀 How to Use Image Generation

### Method 1: Direct API Call (Recommended)

```python
import asyncio
from whiteboard_pipeline.components.imagen_client import ImagenClient

async def generate_image():
    # Configure
    config = {
        'google_cloud_project_id': 'your-project-id',
        'google_cloud_location': 'us-central1',
        'imagen_model_version': 'imagegeneration@006'
    }

    # Initialize client
    client = ImagenClient(config)

    # Generate image from text
    image = await client.generate_diagram_image(
        description="User login flowchart with authentication",
        style="professional diagram"
    )

    # Save image
    await client.save_image(image, "output.png")
    print("Image saved!")

# Run
asyncio.run(generate_image())
```

### Method 2: Use the Example Script

We've provided a comprehensive example script:

```bash
python example_imagen_generation.py
```

This script includes 4 examples:
1. Simple diagram generation
2. Flowchart from description
3. Image from Mermaid code
4. Custom style diagram

### Method 3: Run Test 7

```bash
python test_gemini_integration.py
```

Test 7 will attempt to generate an image (will skip if not configured).

---

## 📋 Setup Requirements

### 1. Install Dependencies

```bash
pip install google-cloud-aiplatform
```

### 2. Set Up Google Cloud

1. **Create Google Cloud Project**
   - Go to https://console.cloud.google.com/
   - Create a new project or use existing

2. **Enable Vertex AI API**
   - Navigate to "APIs & Services"
   - Enable "Vertex AI API"

3. **Create Service Account**
   - Go to "IAM & Admin" → "Service Accounts"
   - Create service account with "Vertex AI User" role
   - Download JSON credentials

4. **Configure Authentication**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

5. **Update config.json**
   ```json
   {
       "image_generator": {
           "provider": "imagen",
           "google_cloud_project_id": "your-actual-project-id",
           "google_cloud_location": "us-central1",
           "imagen_model_version": "imagegeneration@006"
       }
   }
   ```

---

## 💡 Usage Examples

### Example 1: Simple Flowchart

```python
from whiteboard_pipeline.components.imagen_client import ImagenClient

client = ImagenClient(config)

# Generate flowchart
image = await client.generate_flowchart_image(
    description="User authentication process",
    flow_direction="TD"
)

await client.save_image(image, "auth_flowchart.png")
```

### Example 2: From Mermaid Code

```python
mermaid_code = """
flowchart TD
    A([Start]) --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[Action]
    C -->|No| E[End]
"""

image = await client.generate_from_mermaid(mermaid_code)
await client.save_image(image, "from_mermaid.png")
```

### Example 3: Custom Style

```python
image = await client.generate_diagram_image(
    description="Software deployment pipeline",
    style="modern technical illustration, colorful, clean"
)

await client.save_image(image, "deployment.png")
```

---

## 💰 Pricing

**Google Imagen (Vertex AI) Pricing:**
- 512x512 images: ~$0.020 per image
- 1024x1024 images: ~$0.040 per image

**Free Tier:**
- Google Cloud offers $300 credit for new users
- Can generate ~7,500-15,000 images with free credit

**Example costs:**
- 100 images/month: ~$2-4
- 1000 images/month: ~$20-40

See: https://cloud.google.com/vertex-ai/pricing

---

## 🔍 What Works vs What Doesn't

### ✅ What Works:

1. **Direct API calls** - Use ImagenClient directly
   ```python
   client = ImagenClient(config)
   image = await client.generate_diagram_image("...")
   ```

2. **Standalone scripts** - Run `example_imagen_generation.py`

3. **Manual integration** - Call from your own code

### ❌ What Doesn't Work (Yet):

1. **Automatic pipeline integration** - Can't do this yet:
   ```python
   # This DOESN'T work yet
   result = await pipeline.process(input_data, generate_image=True)
   ```

2. **Combined Mermaid + Image** - Pipeline doesn't automatically generate both

3. **CLI command** - No command-line interface yet

---

## 🚧 Future Integration Plan

**To integrate into pipeline (Phase 3):**

1. Add `generate_image` parameter to pipeline input
2. Orchestrate Gemini + Imagen in sequence
3. Return both Mermaid code and image
4. Add to demo.py examples

**Example of future usage:**
```python
# Future - not implemented yet
result = await pipeline.process(
    input_data,
    output_formats=['mermaid', 'image']  # Get both!
)

print(result.mermaid_code)  # Mermaid syntax
print(result.image_path)     # Generated image
```

---

## 📊 Test Output

When you run `python test_gemini_integration.py`, Test 7 will:

**If Google Cloud NOT configured:**
```
TEST 7: Image Generation (Google Imagen)
⏭️  Skipping: Google Cloud Project not configured
   To enable: [setup instructions]
```

**If Google Cloud IS configured:**
```
TEST 7: Image Generation (Google Imagen)
✓ Imagen client initialized
  Project: your-project-id
  Location: us-central1

Generating image...
✅ Image generation successful!
   Size: (1024, 1024)

💾 Saved to: test_output_imagen.png
```

---

## 🎨 Output Quality

**What to expect:**
- Professional-looking diagrams
- Clean, technical illustration style
- Good prompt following
- Suitable for presentations

**Limitations:**
- May not perfectly match Mermaid syntax
- Text in images may not be perfectly readable
- Better for conceptual diagrams than precise flowcharts

**Recommendation:** Use Mermaid code for precision, images for visual appeal.

---

## ❓ FAQ

**Q: Can I use it without Google Cloud?**
A: No, Imagen requires Google Cloud. Alternative: Use local Stable Diffusion (not implemented).

**Q: Is it free?**
A: No, costs ~$0.02-0.04 per image. But Google offers $300 free credit for new users.

**Q: Why isn't it integrated into the pipeline?**
A: We focused on getting Gemini (text-to-Mermaid) working first. Image generation is Phase 3.

**Q: Can I use a different image generation service?**
A: Yes! You can implement a different client following the same interface. Consider:
- Local Stable Diffusion (free, requires GPU)
- OpenAI DALL-E (similar pricing)
- Midjourney (via API)

**Q: When will it be integrated?**
A: After Phase 2 (sketch-to-flowchart) is complete.

---

## 📚 Resources

- Example script: `example_imagen_generation.py`
- Test: `test_gemini_integration.py` (Test 7)
- Component: `whiteboard_pipeline/components/imagen_client.py`
- Google Imagen docs: https://cloud.google.com/vertex-ai/docs/generative-ai/image/overview
- Vertex AI pricing: https://cloud.google.com/vertex-ai/pricing

---

## 🎯 Summary

**Current State:**
- ✅ ImagenClient fully implemented
- ✅ Standalone usage works
- ✅ Test included
- ❌ Not integrated into pipeline

**To use now:**
1. Set up Google Cloud (one-time)
2. Install `google-cloud-aiplatform`
3. Configure credentials
4. Use directly: `ImagenClient.generate_diagram_image()`

**Future state:**
- Automatic pipeline integration
- Combined Mermaid + Image output
- CLI interface
