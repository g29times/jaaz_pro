"""
Google Imagen Client for Image Generation
Text-to-image generation using Google's Imagen model via Vertex AI
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image

# Google Cloud AI Platform SDK
try:
    from google.cloud import aiplatform
    from vertexai.preview.vision_models import ImageGenerationModel
    IMAGEN_AVAILABLE = True
except ImportError:
    IMAGEN_AVAILABLE = False
    logging.warning("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")


class ImagenClient:
    """Client for Google Imagen API - Text-to-Image Generation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not IMAGEN_AVAILABLE:
            raise ImportError("google-cloud-aiplatform package is required. Install with: pip install google-cloud-aiplatform")

        # Configuration
        self.project_id = config.get('google_cloud_project_id', 'YOUR_GOOGLE_CLOUD_PROJECT_ID')
        self.location = config.get('google_cloud_location', 'us-central1')
        self.model_version = config.get('imagen_model_version', 'imagegeneration@006')

        # Image generation parameters
        self.number_of_images = config.get('number_of_images', 1)
        self.guidance_scale = config.get('guidance_scale', 15)  # How closely to follow prompt (0-20)
        self.add_watermark = config.get('add_watermark', False)

        # Initialize Vertex AI
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            self.logger.info(f"Vertex AI initialized: project={self.project_id}, location={self.location}")

            # Load Imagen model
            self.model = ImageGenerationModel.from_pretrained(self.model_version)
            self.logger.info(f"Imagen model loaded: {self.model_version}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Imagen: {e}")
            self.logger.error("Make sure:")
            self.logger.error("  1. Google Cloud project is set up")
            self.logger.error("  2. Vertex AI API is enabled")
            self.logger.error("  3. GOOGLE_APPLICATION_CREDENTIALS environment variable is set")
            raise

    async def check_health(self) -> Dict[str, Any]:
        """Check if Imagen API is accessible"""
        try:
            # Try a simple generation to test connectivity
            test_prompt = "simple shape"

            result = await asyncio.to_thread(
                self.model.generate_images,
                prompt=test_prompt,
                number_of_images=1
            )

            return {
                'status': 'healthy',
                'api_accessible': True,
                'model': self.model_version,
                'project': self.project_id,
                'images_generated': result is not None
            }

        except Exception as e:
            self.logger.warning(f"Imagen health check failed: {e}")
            return {
                'status': 'unhealthy',
                'api_accessible': False,
                'error': str(e)
            }

    async def generate_diagram_image(self, description: str,
                                    style: str = "professional diagram",
                                    image_size: int = 1024) -> Optional[Image.Image]:
        """
        Generate diagram image from text description

        Args:
            description: Text description of what to generate
            style: Visual style (default: "professional diagram")
            image_size: Image size in pixels (256, 512, 1024)

        Returns:
            PIL Image object or None if generation fails
        """

        prompt = self._create_diagram_prompt(description, style)

        self.logger.info(f"Generating image with Imagen")
        self.logger.debug(f"Prompt: {prompt}")

        try:
            # Generate images (run in thread to avoid blocking)
            images = await asyncio.to_thread(
                self.model.generate_images,
                prompt=prompt,
                number_of_images=self.number_of_images,
                guidance_scale=self.guidance_scale,
                add_watermark=self.add_watermark
            )

            if images and len(images) > 0:
                # Get the first image
                image = images[0]._pil_image

                self.logger.info(f"Image generated successfully: {image.size}")
                return image
            else:
                self.logger.warning("Imagen returned no images")
                return None

        except Exception as e:
            self.logger.error(f"Imagen generation failed: {e}")
            return None

    async def generate_flowchart_image(self, description: str,
                                      flow_direction: str = "TD") -> Optional[Image.Image]:
        """
        Generate flowchart diagram image from description

        Args:
            description: Process description or flowchart details
            flow_direction: Flow direction (TD, LR, etc.) - for context

        Returns:
            PIL Image of generated flowchart
        """

        style = "flowchart diagram"
        flowchart_prompt = f"Flowchart showing {description}, with {flow_direction} flow direction"

        return await self.generate_diagram_image(flowchart_prompt, style)

    async def generate_from_mermaid(self, mermaid_code: str) -> Optional[Image.Image]:
        """
        Generate visual diagram image from Mermaid code

        Args:
            mermaid_code: Mermaid flowchart code

        Returns:
            PIL Image visualizing the flowchart
        """

        # Parse Mermaid code to extract description
        description = self._extract_description_from_mermaid(mermaid_code)

        self.logger.info(f"Generating image from Mermaid code")

        return await self.generate_flowchart_image(description)

    def _create_diagram_prompt(self, description: str, style: str) -> str:
        """Create optimized prompt for diagram generation"""

        # Base prompt template for professional diagrams
        prompt = f"""{style}, {description},
clean lines, simple geometric shapes, white background,
technical illustration, vector art style,
minimalist design, high contrast, clear readable labels,
professional business diagram, infographic style"""

        return prompt.strip()

    def _extract_description_from_mermaid(self, mermaid_code: str) -> str:
        """
        Extract a text description from Mermaid code for image generation

        Args:
            mermaid_code: Mermaid flowchart code

        Returns:
            Text description of the flowchart
        """

        # Simple extraction - get node labels
        import re

        # Extract text from nodes like: A[Start], B[Process], C{Decision}
        node_pattern = r'\w+\[([^\]]+)\]|\w+\{([^\}]+)\}|\w+\(\[([^\]]+)\]\)'
        matches = re.findall(node_pattern, mermaid_code)

        # Flatten and filter empty strings
        labels = [label for match in matches for label in match if label]

        if labels:
            description = " → ".join(labels)
            return description
        else:
            return "process workflow diagram"

    async def save_image(self, image: Image.Image, output_path: Path) -> bool:
        """
        Save generated image to file

        Args:
            image: PIL Image object
            output_path: Path to save the image

        Returns:
            True if saved successfully
        """

        try:
            await asyncio.to_thread(image.save, str(output_path))
            self.logger.info(f"Image saved to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            return False
