"""
Google Gemini Client for LLM Integration
Primary LLM provider with vision capabilities
Uses the new unified Google GenAI SDK
"""

import logging
import asyncio
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# New Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-genai not installed. Run: pip install google-genai")


class GeminiClient:
    """Client for Google Gemini API - PRIMARY LLM provider"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")

        # Configuration
        self.api_key = config.get('gemini_api_key', os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE'))
        self.model_name = config.get('gemini_model', 'gemini-2.5-flash')  # Latest stable model
        self.temperature = config.get('temperature', 0.3)
        self.timeout = config.get('timeout', 60)

        # Initialize client
        # If API key is set in environment variable GEMINI_API_KEY, it will be picked up automatically
        if self.api_key and self.api_key != 'YOUR_GEMINI_API_KEY_HERE':
            self.client = genai.Client(api_key=self.api_key)
        else:
            # Try to use environment variable
            self.client = genai.Client()

        self.logger.info(f"Gemini client initialized with model: {self.model_name}")

    async def check_health(self) -> Dict[str, Any]:
        """Check if Gemini API is accessible"""
        try:
            # Try a simple generation to test connectivity
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents="test"
            )

            return {
                'status': 'healthy',
                'api_accessible': True,
                'model': self.model_name,
                'response_received': response.text is not None
            }
        except Exception as e:
            self.logger.warning(f"Gemini health check failed: {e}")
            return {
                'status': 'unhealthy',
                'api_accessible': False,
                'error': str(e)
            }

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> Optional[str]:
        """Generate text using Gemini"""
        try:
            # Combine system instruction with prompt if provided
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            self.logger.debug(f"Sending request to Gemini ({self.model_name})")

            # Generate content with configuration
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=8192,
            )

            # Generate content (run in thread to avoid blocking)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=full_prompt,
                config=config
            )

            if response and response.text:
                generated_text = response.text.strip()
                self.logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
            else:
                self.logger.warning("Gemini returned empty response")
                return None

        except Exception as e:
            self.logger.error(f"Gemini generation failed: {e}")
            return None

    async def generate_mermaid_from_text(self, content: str, flow_direction: str = "TD") -> Optional[str]:
        """Generate Mermaid flowchart from text description using Gemini"""

        system_instruction = """You are an expert at creating Mermaid flowchart diagrams.
Generate clean, syntactically correct Mermaid flowchart code from process descriptions.
Use appropriate shapes: rectangles for processes, diamonds for decisions, ovals for start/end.
Keep node labels concise and meaningful.
IMPORTANT: Output ONLY the Mermaid code, no explanations or markdown formatting."""

        user_prompt = f"""Convert this process description into a Mermaid flowchart with {flow_direction} direction:

{content}

Generate ONLY the Mermaid flowchart code. Start with 'flowchart {flow_direction}'.

Example format:
flowchart TD
    A([Start]) --> B[Process Step]
    B --> C{{Decision?}}
    C -->|Yes| D[Action]
    C -->|No| E[Alternative]
    D --> F([End])
    E --> F

Now generate the flowchart for the given process description:"""

        self.logger.info(f"Generating Mermaid flowchart using Gemini {self.model_name}")
        result = await self.generate(user_prompt, system_instruction)

        if result:
            # Clean up the response - extract just the Mermaid code
            result = self._extract_mermaid_code(result)

        return result

    async def generate_mermaid_from_image(self, image_path: str, flow_direction: str = "TD") -> Optional[str]:
        """
        Generate Mermaid flowchart from image file using Gemini's vision capabilities

        Args:
            image_path: Path to image file
            flow_direction: Flow direction (TD, LR, etc.)

        Returns:
            Mermaid flowchart code or None
        """
        try:
            from PIL import Image

            # Load image
            image = Image.open(image_path)
            self.logger.info(f"Loaded image from: {image_path}")

            # Use the PIL Image method
            return await self.generate_mermaid_from_image_object(image, flow_direction)

        except Exception as e:
            self.logger.error(f"Failed to load image from {image_path}: {e}")
            return None

    async def generate_mermaid_from_image_object(self, image: "Image.Image", flow_direction: str = "TD") -> Optional[str]:
        """
        Generate Mermaid flowchart from PIL Image object using Gemini's vision capabilities

        Args:
            image: PIL Image object
            flow_direction: Flow direction (TD, LR, etc.)

        Returns:
            Mermaid flowchart code or None
        """
        try:
            system_instruction = """You are an expert at analyzing hand-drawn flowcharts and diagrams.
Analyze the image and convert it into a clean Mermaid flowchart diagram.
Identify shapes, text labels, arrows, and connections.
Generate syntactically correct Mermaid code.
IMPORTANT: Output ONLY the Mermaid code, no explanations."""

            prompt = f"""Analyze this hand-drawn flowchart image and convert it to Mermaid code with {flow_direction} direction.

Identify:
1. All shapes (rectangles, diamonds, ovals)
2. Text labels within shapes
3. Arrows and connections between shapes
4. Flow direction and logic

Generate ONLY the Mermaid flowchart code starting with 'flowchart {flow_direction}'."""

            # Combine system instruction and prompt
            full_prompt = f"{system_instruction}\n\n{prompt}"

            self.logger.info(f"Analyzing image with Gemini Vision: {image.size}")

            # Convert PIL image to bytes
            import io
            img_byte_arr = io.BytesIO()
            # Always save as PNG for consistency
            image_format = image.format if image.format else 'PNG'
            if image_format.upper() not in ['PNG', 'JPEG', 'JPG']:
                image_format = 'PNG'
            image.save(img_byte_arr, format=image_format)
            img_byte_arr = img_byte_arr.getvalue()

            mime_type = f"image/{image_format.lower()}"
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            self.logger.debug(f"Image converted to {mime_type}, size: {len(img_byte_arr)} bytes")

            # Generate with image using new SDK
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=8192,
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[
                    types.Part.from_text(full_prompt),
                    types.Part.from_bytes(data=img_byte_arr, mime_type=mime_type)
                ],
                config=config
            )

            if response and response.text:
                result = response.text.strip()
                result = self._extract_mermaid_code(result)
                self.logger.info(f"Generated Mermaid code from image ({len(result)} chars)")
                return result
            else:
                self.logger.warning("Gemini vision returned empty response")
                return None

        except Exception as e:
            self.logger.error(f"Gemini vision generation failed: {e}")
            return None

    def _extract_mermaid_code(self, text: str) -> str:
        """Extract clean Mermaid code from LLM response"""
        # Remove markdown code blocks if present
        if '```mermaid' in text:
            start = text.find('```mermaid') + 10
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()

        # Ensure it starts with flowchart
        lines = text.strip().split('\n')
        if lines and not lines[0].strip().startswith('flowchart'):
            # Look for flowchart line
            for i, line in enumerate(lines):
                if line.strip().startswith('flowchart'):
                    return '\n'.join(lines[i:])

        return text.strip()

    async def generate_mermaid_from_elements(self, elements: List[Any],
                                            context: Dict[str, Any]) -> Optional[str]:
        """Generate Mermaid flowchart from parsed visual elements using Gemini"""

        system_instruction = """You are an expert at creating Mermaid flowchart diagrams from visual element descriptions.
Generate clean, syntactically correct Mermaid flowchart code that represents the spatial relationships and flow logic.
IMPORTANT: Output ONLY the Mermaid code, no explanations."""

        # Build description of elements
        element_descriptions = []
        for i, elem in enumerate(elements[:20]):  # Limit to first 20 elements
            # Handle both dict and ParsedElement objects
            if hasattr(elem, 'element_type'):
                elem_type = elem.element_type
                content = elem.content
            else:
                elem_type = elem.get('element_type', 'unknown')
                content = elem.get('content', '')

            element_descriptions.append(f"{i+1}. {elem_type}: {content}")

        elements_text = '\n'.join(element_descriptions)

        user_prompt = f"""Create a Mermaid flowchart from these detected visual elements:

{elements_text}

Flow direction: {context.get('flow_direction', 'TD')}

Generate ONLY the Mermaid flowchart code. Use appropriate shapes based on the element types.
Start with 'flowchart {context.get('flow_direction', 'TD')}'.

Now generate the flowchart:"""

        self.logger.info(f"Generating Mermaid from {len(elements)} elements using Gemini")
        result = await self.generate(user_prompt, system_instruction)

        if result:
            result = self._extract_mermaid_code(result)

        return result

    async def generate_image_from_text(self, prompt: str, aspect_ratio: str = "1:1") -> Optional[bytes]:
        """
        Generate image from text prompt using Gemini's native image generation capability

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Image aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", etc.)

        Returns:
            Image bytes (PNG format) or None if generation fails
        """
        self.logger.info(f"Generating image with Gemini (aspect ratio: {aspect_ratio})")

        try:
            # Use Gemini 2.5 Flash Image model for native image generation
            config = types.GenerateContentConfig(
                temperature=1.0,  # Higher temperature for creative image generation
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio
                )
            )

            # Generate image using Gemini 2.5 Flash Image
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model="gemini-2.5-flash-image",  # Native image generation model
                contents=prompt,
                config=config
            )

            # Extract image from response parts
            if response and hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Return the image bytes directly
                        self.logger.info("Image generated successfully")
                        return part.inline_data.data

            self.logger.warning("No image data found in response")
            return None

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None

    async def generate_diagram_image(self, description: str, style: str = "professional flowchart diagram") -> Optional[bytes]:
        """
        Generate diagram/flowchart image from description using Gemini

        Args:
            description: Description of the diagram/flowchart
            style: Visual style for the diagram

        Returns:
            Image bytes or None if generation fails
        """
        prompt = f"""Create a {style}: {description}

Style requirements:
- Clean, professional appearance
- Clear shapes and connections
- Readable text labels
- White or light background
- Technical illustration style
- High contrast for clarity"""

        return await self.generate_image_from_text(prompt)
