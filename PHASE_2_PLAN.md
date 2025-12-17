# Phase 2: Image Sketch → Mermaid Implementation Plan

**Status**: Ready to Implement
**Priority**: HIGH
**Timeline**: 1-2 weeks
**Architecture**: Google Gemini Vision (Primary)

---

## 🎯 Objective

Enable the pipeline to process whiteboard photos and hand-drawn sketches, converting them into clean Mermaid flowcharts using Gemini Vision API.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Whiteboard Photo                   │
│            (PNG, JPG, PDF, hand-drawn sketch)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   1. Image Preprocessing                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ - Load image (PIL)                                    │  │
│  │ - Rotation correction (detect & fix skew)            │  │
│  │ - Noise reduction (denoise)                          │  │
│  │ - Contrast enhancement (CLAHE)                       │  │
│  │ - Resize/normalize (optimal size for Gemini)        │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               2. Gemini Vision Analysis ⭐                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Model: gemini-2.5-flash (with vision)                │  │
│  │                                                       │  │
│  │ Prompt:                                              │  │
│  │ "Analyze this hand-drawn flowchart diagram and       │  │
│  │  convert it to Mermaid code. Identify:              │  │
│  │  - All shapes (rectangles, diamonds, ovals)         │  │
│  │  - Text labels within shapes                        │  │
│  │  - Arrows and connections                           │  │
│  │  - Flow direction and logic"                        │  │
│  │                                                       │  │
│  │ Output: Mermaid flowchart code                       │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ├─── Success ──────┐
                       │                  │
                       ├─── Failure ───┐  │
                       │               │  │
                       ▼               ▼  ▼
┌──────────────────────────────┐   ┌────────────────────────┐
│  3a. Fallback: qwen2.5vl     │   │ 4. Post-Processing     │
│  (Ollama - local vision LM)  │   │ - Validate Mermaid     │
│                              │   │ - Clean up syntax      │
│  Same prompt, local inference│   │ - Ensure flowchart TD  │
└──────────────────────────────┘   └───────────┬────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │   Clean Mermaid Code  │
                                    │   Ready for rendering │
                                    └──────────────────────┘
```

---

## 📦 Components to Implement/Update

### 1. ImageInputHandler (NEW)

**File**: `whiteboard_pipeline/components/image_input_handler.py`

**Purpose**: Handle loading and preprocessing of images

```python
class ImageInputHandler:
    """Handle image input and preprocessing for sketch-to-flowchart conversion"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Preprocessing parameters
        self.target_size = config.get('image_target_size', (1024, 1024))
        self.enable_rotation_correction = config.get('enable_rotation_correction', True)
        self.enable_contrast_enhancement = config.get('enable_contrast_enhancement', True)

    async def load_and_preprocess(self, input_path: str) -> Image:
        """Load image and apply preprocessing pipeline"""
        # Load image
        image = self._load_image(input_path)

        # Preprocessing steps
        if self.enable_rotation_correction:
            image = self._correct_rotation(image)

        if self.enable_contrast_enhancement:
            image = self._enhance_contrast(image)

        # Resize to optimal size
        image = self._resize_for_vision_api(image)

        return image

    def _load_image(self, path: str) -> Image:
        """Load image from file path"""
        # Support PNG, JPG, PDF
        pass

    def _correct_rotation(self, image: Image) -> Image:
        """Detect and correct image rotation"""
        # Use OpenCV to detect text orientation
        # Rotate image to correct orientation
        pass

    def _enhance_contrast(self, image: Image) -> Image:
        """Enhance image contrast for better recognition"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        pass

    def _resize_for_vision_api(self, image: Image) -> Image:
        """Resize image to optimal size for Gemini Vision"""
        # Maintain aspect ratio
        # Target: 1024x1024 max
        pass
```

**Status**: ⚠️ Needs implementation

---

### 2. GeminiClient (UPDATE)

**File**: `whiteboard_pipeline/components/gemini_client.py`

**Current**: Already has `generate_mermaid_from_image()` ✅

**Enhancement Needed**:
- Update prompts for better sketch understanding
- Add support for image bytes (not just file paths)
- Improve error handling for vision API

```python
# Current implementation
async def generate_mermaid_from_image(self, image_path: str, flow_direction: str = "TD") -> Optional[str]:
    """Generate Mermaid flowchart from image using Gemini's vision capabilities"""
    # Already implemented! ✅

# Enhancement: Support PIL Image objects
async def generate_mermaid_from_image_object(self, image: Image, flow_direction: str = "TD") -> Optional[str]:
    """Generate Mermaid from PIL Image object"""
    # Convert PIL Image to bytes
    # Call Gemini Vision API
    pass
```

**Status**: ✅ Core method exists, minor enhancements needed

---

### 3. SimpleSketchToMermaidPipeline (UPDATE)

**File**: `whiteboard_pipeline/simple_pipeline.py`

**Update**: Add IMAGE input type handling

```python
class SimpleSketchToMermaidPipeline:
    def __init__(self, config_path: str = "config.json"):
        # ... existing initialization ...

        # ADD: Image input handler
        self.image_handler = ImageInputHandler(self.config.get('image_input', {}))

    async def process_sketch_to_mermaid(self, user_input: WhiteboardInput) -> PipelineResult:
        """Main processing method"""

        # ... existing code ...

        # ADD: Handle IMAGE input type
        if user_input.input_type == InputType.IMAGE:
            return await self._process_image_input(user_input)

        # ... existing TEXT handling ...

    async def _process_image_input(self, user_input: WhiteboardInput) -> PipelineResult:
        """Process image/sketch input"""
        try:
            # 1. Load and preprocess image
            image = await self.image_handler.load_and_preprocess(user_input.image_path)

            # 2. Generate Mermaid using Gemini Vision
            mermaid_code = await self.gemini_client.generate_mermaid_from_image_object(
                image=image,
                flow_direction=user_input.parameters.get('direction', 'TD')
            )

            if not mermaid_code:
                # Fallback to qwen2.5vl if available
                if self.ollama_available:
                    mermaid_code = await self.ollama_client.generate_mermaid_from_image(image)

            # 3. Create result
            output = GeneratedOutput(
                content=mermaid_code,
                format='mermaid',
                metadata={
                    'generator': 'gemini_vision',
                    'source': 'image',
                    'preprocessing': 'applied'
                }
            )

            return PipelineResult(
                success=True,
                outputs=[output],
                execution_time=time.time() - start_time
            )

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return PipelineResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
```

**Status**: ⚠️ Needs implementation

---

## 🧪 Testing Strategy

### Test Suite: `test_image_to_mermaid.py` (NEW)

```python
"""
Tests for Image Sketch → Mermaid conversion
"""

async def test_1_load_and_preprocess():
    """Test image loading and preprocessing"""
    # Test loading PNG, JPG, PDF
    # Test rotation correction
    # Test contrast enhancement
    pass

async def test_2_simple_sketch():
    """Test simple hand-drawn flowchart"""
    # Input: Simple sketch (Start → Process → End)
    # Expected: Valid Mermaid with 3 nodes
    pass

async def test_3_complex_sketch():
    """Test complex flowchart with decisions"""
    # Input: Sketch with diamonds, multiple branches
    # Expected: Valid Mermaid with decision nodes
    pass

async def test_4_rotated_image():
    """Test rotated/skewed whiteboard photo"""
    # Input: Rotated image
    # Expected: Correct rotation detection and processing
    pass

async def test_5_noisy_image():
    """Test low-quality or noisy photo"""
    # Input: Photo with noise, poor lighting
    # Expected: Still generates valid Mermaid
    pass

async def test_6_fallback_to_ollama():
    """Test fallback to qwen2.5vl when Gemini fails"""
    # Simulate Gemini failure
    # Expected: Ollama fallback works
    pass

async def test_7_end_to_end():
    """Test full pipeline with real whiteboard photo"""
    # Input: Real whiteboard photo
    # Expected: Clean Mermaid flowchart
    pass
```

---

## 📊 Test Data Needed

### Sample Images to Create

1. **Simple Flowchart** (PNG)
   - 3-4 boxes with arrows
   - Clear labels
   - Straight capture

2. **Complex Flowchart** (PNG)
   - 7-10 nodes
   - Decision diamonds
   - Multiple branches

3. **Rotated Photo** (JPG)
   - Whiteboard photo at 15-degree angle
   - Test rotation correction

4. **Noisy Photo** (JPG)
   - Poor lighting
   - Shadow artifacts
   - Test preprocessing

5. **Hand-Drawn Sketch** (PNG)
   - Irregular shapes
   - Handwritten text
   - Test OCR and understanding

**Location**: `test_images/sketches/`

---

## 🔧 Configuration

### Update config.json

```json
{
    "image_input": {
        "image_target_size": [1024, 1024],
        "enable_rotation_correction": true,
        "enable_contrast_enhancement": true,
        "enable_noise_reduction": true,
        "preprocessing_level": "auto"
    },
    "mermaid_generator": {
        "llm_provider": "gemini",
        "gemini_api_key": "YOUR_API_KEY",
        "gemini_model": "gemini-2.5-flash",
        "enable_vision": true,
        "vision_fallback_to_ollama": true,
        "ollama_vision_model": "qwen2.5vl:latest"
    }
}
```

---

## 📝 Implementation Checklist

### Week 1: Core Implementation

- [ ] **Day 1-2: ImageInputHandler**
  - [ ] Create `image_input_handler.py`
  - [ ] Implement `load_and_preprocess()`
  - [ ] Implement rotation correction
  - [ ] Implement contrast enhancement
  - [ ] Add unit tests

- [ ] **Day 3-4: Pipeline Integration**
  - [ ] Update `simple_pipeline.py`
  - [ ] Add `_process_image_input()` method
  - [ ] Wire up ImageInputHandler
  - [ ] Add error handling
  - [ ] Test integration

- [ ] **Day 5: GeminiClient Enhancement**
  - [ ] Add `generate_mermaid_from_image_object()`
  - [ ] Update vision prompts for better sketch understanding
  - [ ] Improve error handling
  - [ ] Test vision API calls

### Week 2: Testing & Polish

- [ ] **Day 1-2: Test Suite**
  - [ ] Create `test_image_to_mermaid.py`
  - [ ] Implement 7 test cases
  - [ ] Create sample test images
  - [ ] Run all tests

- [ ] **Day 3-4: Demo & Documentation**
  - [ ] Add image examples to `demo.py`
  - [ ] Create sample whiteboard photos
  - [ ] Update `QUICK_START.md`
  - [ ] Update `README.md`

- [ ] **Day 5: Optimization**
  - [ ] Benchmark performance
  - [ ] Optimize preprocessing
  - [ ] Add caching if needed
  - [ ] Final testing

---

## 🎯 Success Criteria

### Technical Goals
- ✅ Pipeline accepts IMAGE input type
- ✅ Gemini Vision successfully analyzes sketches
- ✅ 90%+ accuracy on clean whiteboard photos
- ✅ Handles rotated images (up to 30 degrees)
- ✅ Processing time < 5 seconds
- ✅ Fallback to Ollama works
- ✅ 7/7 tests passing

### User Experience Goals
- ✅ Simple API: `pipeline.process_sketch_to_mermaid(image_input)`
- ✅ Clear error messages
- ✅ Works with phone photos
- ✅ Handles various image qualities
- ✅ Produces clean, valid Mermaid code

---

## 🚀 Example Usage (Target)

```python
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType

# Initialize pipeline
pipeline = SimpleSketchToMermaidPipeline()

# Process whiteboard photo
input_data = WhiteboardInput(
    input_type=InputType.IMAGE,
    image_path="whiteboard_photo.jpg",
    parameters={'direction': 'TD'}
)

# Generate Mermaid code
result = await pipeline.process_sketch_to_mermaid(input_data)

if result.success:
    print(result.outputs[0].content)  # Mermaid code
    # Save or render the flowchart
else:
    print(f"Error: {result.error_message}")
```

---

## 💡 Key Technical Decisions

### 1. Gemini Vision as Primary
**Why**:
- Best-in-class vision understanding
- Same API key as text generation
- No need for complex CV pipelines
- Handles handwriting well

### 2. Minimal Preprocessing
**Why**:
- Gemini Vision is robust to noise and rotation
- Focus on essential preprocessing only
- Faster processing
- Simpler codebase

### 3. Ollama qwen2.5vl as Fallback
**Why**:
- Offline capability
- Free local inference
- Good vision understanding
- Already integrated

---

## 📚 Resources

### Gemini Vision API
- Docs: https://ai.google.dev/gemini-api/docs/vision
- Image Understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Best Practices: https://ai.google.dev/gemini-api/docs/vision#best-practices

### Image Preprocessing
- OpenCV Python: https://docs.opencv.org/4.x/
- PIL/Pillow: https://pillow.readthedocs.io/
- CLAHE: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

---

## 🎉 Outcome

After Phase 2 completion:
- ✅ Users can take whiteboard photos with their phone
- ✅ Pipeline converts photos to clean Mermaid flowcharts
- ✅ Works with hand-drawn sketches
- ✅ Handles real-world image quality issues
- ✅ Full Google AI stack: Text → Mermaid, Image → Mermaid, Text → Image

**Ready to build! 🚀**
