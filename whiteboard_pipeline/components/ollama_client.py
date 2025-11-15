"""
Ollama Client for Local LLM Integration
Provides local, cost-free LLM inference for development and testing
"""

import logging
import json
from typing import Dict, Any, Optional, List
import aiohttp


class OllamaClient:
    """Client for interacting with local Ollama server"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Ollama configuration
        self.base_url = config.get('ollama_url', 'http://localhost:11434')
        self.model = config.get('ollama_model', 'llama3.2')  # Default to smallest model
        self.temperature = config.get('temperature', 0.7)
        self.timeout = config.get('timeout', 60)

        self.logger.info(f"Ollama client initialized with model: {self.model}")

    async def check_health(self) -> Dict[str, Any]:
        """Check if Ollama server is running and model is available"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check server health
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]

                        return {
                            'status': 'healthy',
                            'server_running': True,
                            'models_available': models,
                            'requested_model': self.model,
                            'model_ready': any(self.model in m for m in models)
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'server_running': False,
                            'error': f"Server returned {response.status}"
                        }
        except Exception as e:
            self.logger.warning(f"Ollama health check failed: {e}")
            return {
                'status': 'unhealthy',
                'server_running': False,
                'error': str(e)
            }

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate text using Ollama"""
        try:
            # Build request payload
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': self.temperature,
                }
            }

            if system_prompt:
                payload['system'] = system_prompt

            self.logger.debug(f"Sending request to Ollama with model {self.model}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Ollama request failed: {response.status} - {error_text}")
                        return None

                    data = await response.json()
                    generated_text = data.get('response', '').strip()

                    if generated_text:
                        self.logger.debug(f"Generated {len(generated_text)} characters")
                        return generated_text
                    else:
                        self.logger.warning("Ollama returned empty response")
                        return None

        except asyncio.TimeoutError:
            self.logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return None

    async def generate_mermaid_from_text(self, content: str, flow_direction: str = "TD") -> Optional[str]:
        """Generate Mermaid flowchart from text description"""

        system_prompt = """You are an expert at creating Mermaid flowchart diagrams.
Generate clean, syntactically correct Mermaid flowchart code from process descriptions.
Use appropriate shapes: rectangles for processes, diamonds for decisions, ovals for start/end.
Keep node labels concise and meaningful."""

        user_prompt = f"""Convert this process description into a Mermaid flowchart with {flow_direction} direction:

{content}

Generate ONLY the Mermaid flowchart code, nothing else. Start with 'flowchart {flow_direction}'.
Example format:
flowchart TD
    A([Start]) --> B[Process Step]
    B --> C{{Decision?}}
    C -->|Yes| D[Action]
    C -->|No| E[Alternative]
    D --> F([End])
    E --> F

Now generate the flowchart:"""

        self.logger.info(f"Generating Mermaid flowchart using {self.model}")
        result = await self.generate(user_prompt, system_prompt)

        if result:
            # Clean up the response - extract just the Mermaid code
            result = self._extract_mermaid_code(result)

        return result

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

    async def generate_mermaid_from_elements(self, elements: List[Dict[str, Any]],
                                            context: Dict[str, Any]) -> Optional[str]:
        """Generate Mermaid flowchart from parsed visual elements"""

        system_prompt = """You are an expert at creating Mermaid flowchart diagrams from visual element descriptions.
Generate clean, syntactically correct Mermaid flowchart code that represents the spatial relationships and flow logic."""

        # Build description of elements
        element_descriptions = []
        for i, elem in enumerate(elements[:20]):  # Limit to first 20 elements
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

        self.logger.info(f"Generating Mermaid from {len(elements)} elements using {self.model}")
        result = await self.generate(user_prompt, system_prompt)

        if result:
            result = self._extract_mermaid_code(result)

        return result


# Import asyncio for timeout handling
import asyncio
