import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..interfaces import TaskRouterInterface
from ..models import SemanticIntent, TaskPlan, TaskStep, GeneratorType


class TaskRouter(TaskRouterInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm_provider = self.config.get('llm_provider', 'openai')  # openai, anthropic
        self.model_name = self.config.get('model_name', 'gpt-4')
        self.api_key = self.config.get('api_key')
        self.max_tokens = self.config.get('max_tokens', 1500)
        self.temperature = self.config.get('temperature', 0.2)
        
        self.task_templates = {
            'create_flowchart': self._flowchart_template,
            'generate_report': self._report_template,
            'generate_image': self._image_template,
            'analyze_content': self._analysis_template
        }
    
    async def create_task_plan(self, intent: SemanticIntent) -> TaskPlan:
        self.logger.info(f"Creating task plan for intent: {intent.intent}")
        
        prompt = self._build_planning_prompt(intent)
        
        try:
            llm_response = await self._call_llm_api(prompt)
            task_plan_data = self._parse_llm_response(llm_response)
            
            steps = [
                TaskStep(
                    action=step['action'],
                    generator_type=GeneratorType(step['generator_type']),
                    parameters=step['parameters'],
                    dependencies=step.get('dependencies', [])
                )
                for step in task_plan_data['steps']
            ]
            
            return TaskPlan(
                steps=steps,
                priority=task_plan_data.get('priority', 5),
                estimated_time=task_plan_data.get('estimated_time'),
                metadata={
                    'intent': intent.intent,
                    'confidence': intent.confidence,
                    'created_at': datetime.now().isoformat(),
                    'llm_model': self.model_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Task planning failed: {e}")
            return self._fallback_task_plan(intent)
    
    def _build_planning_prompt(self, intent: SemanticIntent) -> str:
        base_prompt = f"""
You are a task planning AI for a whiteboard processing pipeline. Create a detailed execution plan.

Intent Analysis:
- Primary Intent: {intent.intent}
- Confidence: {intent.confidence}
- Context: {json.dumps(intent.context, indent=2)}
- Suggested Tasks: {intent.suggested_tasks}

Available Generators:
1. diffusion_image - Creates images using SDXL (parameters: prompt, style, dimensions)
2. mermaid_flow - Creates flowcharts as .mmd files (parameters: content, direction, theme)
3. report_summarizer - Creates reports as .md/.pdf (parameters: content, format, template)

Create a JSON task plan with this structure:
{{
    "steps": [
        {{
            "action": "descriptive_action_name",
            "generator_type": "diffusion_image|mermaid_flow|report_summarizer",
            "parameters": {{
                "param1": "value1",
                "param2": "value2"
            }},
            "dependencies": ["previous_step_action"] // optional
        }}
    ],
    "priority": 1-10, // higher = more important
    "estimated_time": 30.5 // seconds
}}

"""
        
        template_fn = self.task_templates.get(intent.intent, self._default_template)
        specific_guidance = template_fn(intent)
        
        return base_prompt + specific_guidance
    
    def _flowchart_template(self, intent: SemanticIntent) -> str:
        return """
For flowchart creation:
- Use mermaid_flow generator
- Consider the context complexity for flowchart direction (TD, LR, etc.)
- Include clear node descriptions and connections
- Add styling based on domain (business, technical, etc.)

Example parameters:
{
    "content": "extracted flowchart content",
    "direction": "TD",
    "theme": "default",
    "include_styling": true
}
"""
    
    def _report_template(self, intent: SemanticIntent) -> str:
        return """
For report generation:
- Use report_summarizer generator
- Choose format based on complexity (md for simple, pdf for formal)
- Include executive summary for longer reports
- Structure content logically

Example parameters:
{
    "content": "source content to summarize",
    "format": "md",
    "template": "standard",
    "include_summary": true
}
"""
    
    def _image_template(self, intent: SemanticIntent) -> str:
        return """
For image generation:
- Use diffusion_image generator
- Create descriptive prompts based on whiteboard content
- Choose appropriate style and dimensions
- Consider the domain for style selection

Example parameters:
{
    "prompt": "detailed image description",
    "style": "professional",
    "dimensions": "1024x1024",
    "enhance_quality": true
}
"""
    
    def _analysis_template(self, intent: SemanticIntent) -> str:
        return """
For content analysis:
- Use report_summarizer generator for analysis output
- Focus on key insights and patterns
- Provide actionable recommendations
- Structure analysis clearly

Example parameters:
{
    "content": "content to analyze",
    "format": "md",
    "template": "analysis",
    "include_recommendations": true
}
"""
    
    def _default_template(self, intent: SemanticIntent) -> str:
        return """
For general tasks:
- Choose the most appropriate generator based on intent
- Keep parameters simple and focused
- Ensure the output meets the user's likely needs
"""
    
    async def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API (OpenAI or Anthropic)"""
        
        if self.llm_provider == 'openai':
            return await self._call_openai_api(prompt)
        elif self.llm_provider == 'anthropic':
            return await self._call_anthropic_api(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    async def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a task planning AI. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
    
    async def _call_anthropic_api(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")
    
    def _parse_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response from LLM API"""
        try:
            if self.llm_provider == 'openai':
                content = response['choices'][0]['message']['content']
            else:  # anthropic
                content = response['content'][0]['text']
            
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                return json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")
                
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    def _fallback_task_plan(self, intent: SemanticIntent) -> TaskPlan:
        """Create a simple fallback task plan when LLM fails"""
        self.logger.info("Using fallback task planning")
        
        if intent.intent == 'create_flowchart':
            steps = [
                TaskStep(
                    action="generate_flowchart",
                    generator_type=GeneratorType.MERMAID_FLOW,
                    parameters={
                        "content": "Create flowchart from whiteboard content",
                        "direction": "TD",
                        "theme": "default"
                    }
                )
            ]
        elif intent.intent == 'generate_report':
            steps = [
                TaskStep(
                    action="create_report",
                    generator_type=GeneratorType.REPORT_SUMMARIZER,
                    parameters={
                        "content": "Summarize whiteboard content",
                        "format": "md",
                        "template": "standard"
                    }
                )
            ]
        elif intent.intent == 'generate_image':
            steps = [
                TaskStep(
                    action="generate_image",
                    generator_type=GeneratorType.DIFFUSION_IMAGE,
                    parameters={
                        "prompt": "Create image based on whiteboard content",
                        "style": "professional",
                        "dimensions": "1024x1024"
                    }
                )
            ]
        else:
            steps = [
                TaskStep(
                    action="analyze_content",
                    generator_type=GeneratorType.REPORT_SUMMARIZER,
                    parameters={
                        "content": "Analyze and explain whiteboard content",
                        "format": "md",
                        "template": "analysis"
                    }
                )
            ]
        
        return TaskPlan(
            steps=steps,
            priority=5,
            estimated_time=30.0,
            metadata={
                'intent': intent.intent,
                'fallback': True,
                'created_at': datetime.now().isoformat()
            }
        )