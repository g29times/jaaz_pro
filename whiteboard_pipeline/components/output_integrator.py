import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile

from ..interfaces import OutputIntegratorInterface
from ..models import GeneratorOutput, ProcessingResult, WhiteboardInput


class OutputIntegrator(OutputIntegratorInterface):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.output_directory = Path(self.config.get('output_directory', tempfile.gettempdir()))
        self.create_archive = self.config.get('create_archive', True)
        self.feedback_enabled = self.config.get('feedback_enabled', True)
        self.attachment_formats = self.config.get('attachment_formats', ['png', 'mmd', 'md', 'pdf'])
        
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    async def integrate_outputs(
        self, 
        outputs: List[GeneratorOutput], 
        original_input: WhiteboardInput
    ) -> ProcessingResult:
        """Integrate all generator outputs and prepare final result"""
        
        self.logger.info(f"Integrating {len(outputs)} outputs")
        
        start_time = datetime.now()
        session_id = self._generate_session_id()
        
        try:
            organized_outputs = await self._organize_outputs(outputs, session_id)
            
            whiteboard_attachments = await self._create_whiteboard_attachments(
                organized_outputs, original_input, session_id
            )
            
            feedback_data = await self._prepare_feedback_collection(
                organized_outputs, original_input
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                outputs=organized_outputs,
                execution_time=execution_time,
                success=True,
                feedback_data=feedback_data
            )
            
            await self._log_processing_result(result, session_id)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Output integration failed: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                outputs=outputs,
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"wb_session_{timestamp}"
    
    async def _organize_outputs(
        self, 
        outputs: List[GeneratorOutput], 
        session_id: str
    ) -> List[GeneratorOutput]:
        """Organize outputs into session directory with proper naming"""
        
        session_dir = self.output_directory / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        organized_outputs = []
        type_counters = {}
        
        for output in outputs:
            if output.metadata and output.metadata.get('error'):
                organized_outputs.append(output)
                continue
            
            output_type = output.output_type
            counter = type_counters.get(output_type, 0) + 1
            type_counters[output_type] = counter
            
            if output.file_path and output.file_path.exists():
                new_filename = f"{output_type}_{counter:02d}{output.file_path.suffix}"
                new_path = session_dir / new_filename
                
                try:
                    shutil.copy2(output.file_path, new_path)
                    
                    organized_output = GeneratorOutput(
                        content=output.content,
                        output_type=output.output_type,
                        file_path=new_path,
                        metadata={
                            **(output.metadata or {}),
                            'session_id': session_id,
                            'organized_path': str(new_path),
                            'original_path': str(output.file_path)
                        }
                    )
                    
                    organized_outputs.append(organized_output)
                    
                except Exception as e:
                    self.logger.error(f"Failed to organize output {output.file_path}: {e}")
                    organized_outputs.append(output)
            else:
                text_filename = f"{output_type}_{counter:02d}.txt"
                text_path = session_dir / text_filename
                
                try:
                    text_path.write_text(str(output.content), encoding='utf-8')
                    
                    organized_output = GeneratorOutput(
                        content=output.content,
                        output_type=output.output_type,
                        file_path=text_path,
                        metadata={
                            **(output.metadata or {}),
                            'session_id': session_id,
                            'organized_path': str(text_path),
                            'content_saved_as_text': True
                        }
                    )
                    
                    organized_outputs.append(organized_output)
                    
                except Exception as e:
                    self.logger.error(f"Failed to save content for {output_type}: {e}")
                    organized_outputs.append(output)
        
        return organized_outputs
    
    async def _create_whiteboard_attachments(
        self,
        outputs: List[GeneratorOutput],
        original_input: WhiteboardInput,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """Create whiteboard-ready attachments from outputs"""
        
        attachments = []
        
        for output in outputs:
            if output.metadata and output.metadata.get('error'):
                continue
            
            if not output.file_path or not output.file_path.exists():
                continue
            
            file_extension = output.file_path.suffix.lower().lstrip('.')
            
            if file_extension in self.attachment_formats:
                attachment = {
                    'type': output.output_type,
                    'file_path': str(output.file_path),
                    'filename': output.file_path.name,
                    'size_bytes': output.file_path.stat().st_size,
                    'created_at': output.metadata.get('created_at', datetime.now().isoformat()),
                    'generator': output.metadata.get('generator', 'unknown'),
                    'session_id': session_id
                }
                
                if output.output_type == 'image':
                    attachment.update({
                        'mime_type': 'image/png',
                        'display_type': 'image',
                        'thumbnail_available': True
                    })
                elif output.output_type == 'mermaid':
                    attachment.update({
                        'mime_type': 'text/plain',
                        'display_type': 'diagram',
                        'render_as': 'mermaid'
                    })
                elif output.output_type in ['md', 'pdf']:
                    attachment.update({
                        'mime_type': 'application/pdf' if file_extension == 'pdf' else 'text/markdown',
                        'display_type': 'document',
                        'preview_available': True
                    })
                
                attachments.append(attachment)
        
        summary_attachment = await self._create_session_summary(outputs, session_id)
        if summary_attachment:
            attachments.append(summary_attachment)
        
        return attachments
    
    async def _create_session_summary(
        self, 
        outputs: List[GeneratorOutput], 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Create a summary document for the processing session"""
        
        try:
            session_dir = self.output_directory / session_id
            summary_path = session_dir / "session_summary.md"
            
            successful_outputs = [o for o in outputs if not (o.metadata and o.metadata.get('error'))]
            error_outputs = [o for o in outputs if o.metadata and o.metadata.get('error')]
            
            summary_content = f"""# Whiteboard Processing Session Summary

**Session ID:** {session_id}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- **Total Outputs:** {len(outputs)}
- **Successful:** {len(successful_outputs)}
- **Errors:** {len(error_outputs)}

## Generated Content

"""
            
            for i, output in enumerate(successful_outputs, 1):
                output_type = output.output_type.title()
                generator = output.metadata.get('generator', 'Unknown') if output.metadata else 'Unknown'
                file_name = output.file_path.name if output.file_path else 'N/A'
                
                summary_content += f"""### {i}. {output_type}
- **Generator:** {generator}
- **File:** {file_name}
- **Created:** {output.metadata.get('created_at', 'Unknown') if output.metadata else 'Unknown'}

"""
            
            if error_outputs:
                summary_content += "## Errors\n\n"
                for i, error_output in enumerate(error_outputs, 1):
                    error_msg = error_output.metadata.get('error_message', 'Unknown error') if error_output.metadata else 'Unknown error'
                    summary_content += f"{i}. {error_msg}\n"
            
            summary_content += f"""
## Files Generated
Total files in session directory: {len(list((session_dir).iterdir()))}

---
*Generated by Whiteboard Processing Pipeline*
"""
            
            summary_path.write_text(summary_content, encoding='utf-8')
            
            return {
                'type': 'summary',
                'file_path': str(summary_path),
                'filename': summary_path.name,
                'size_bytes': summary_path.stat().st_size,
                'created_at': datetime.now().isoformat(),
                'generator': 'output_integrator',
                'session_id': session_id,
                'mime_type': 'text/markdown',
                'display_type': 'document',
                'is_summary': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create session summary: {e}")
            return None
    
    async def _prepare_feedback_collection(
        self,
        outputs: List[GeneratorOutput], 
        original_input: WhiteboardInput
    ) -> Optional[Dict[str, Any]]:
        """Prepare feedback collection data for optimization"""
        
        if not self.feedback_enabled:
            return None
        
        try:
            feedback_data = {
                'session_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'input_type': original_input.input_type.value,
                    'total_outputs': len(outputs),
                    'successful_outputs': len([o for o in outputs if not (o.metadata and o.metadata.get('error'))]),
                    'processing_pipeline_version': '0.1.0'
                },
                'quality_metrics': {
                    'completion_rate': len([o for o in outputs if not (o.metadata and o.metadata.get('error'))]) / len(outputs) if outputs else 0,
                    'average_generation_time': self._calculate_average_generation_time(outputs),
                    'generator_usage': self._calculate_generator_usage(outputs)
                },
                'feedback_questions': [
                    {
                        'id': 'overall_quality',
                        'question': 'How would you rate the overall quality of the generated outputs?',
                        'type': 'rating',
                        'scale': [1, 2, 3, 4, 5],
                        'labels': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
                    },
                    {
                        'id': 'relevance',
                        'question': 'How relevant were the outputs to your original intent?',
                        'type': 'rating',
                        'scale': [1, 2, 3, 4, 5]
                    },
                    {
                        'id': 'improvements',
                        'question': 'What improvements would you suggest?',
                        'type': 'text',
                        'optional': True
                    },
                    {
                        'id': 'preferred_output',
                        'question': 'Which output type was most useful?',
                        'type': 'choice',
                        'options': list(set([o.output_type for o in outputs if not (o.metadata and o.metadata.get('error'))]))
                    }
                ]
            }
            
            return feedback_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare feedback data: {e}")
            return None
    
    def _calculate_average_generation_time(self, outputs: List[GeneratorOutput]) -> float:
        """Calculate average generation time across all outputs"""
        
        times = []
        for output in outputs:
            if output.metadata and 'execution_time' in output.metadata:
                times.append(output.metadata['execution_time'])
        
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_generator_usage(self, outputs: List[GeneratorOutput]) -> Dict[str, int]:
        """Calculate usage statistics for each generator type"""
        
        usage = {}
        for output in outputs:
            if output.metadata and 'generator_type' in output.metadata:
                generator_type = output.metadata['generator_type']
                usage[generator_type] = usage.get(generator_type, 0) + 1
        
        return usage
    
    async def _log_processing_result(self, result: ProcessingResult, session_id: str):
        """Log processing result for monitoring and optimization"""
        
        log_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'success': result.success,
            'execution_time': result.execution_time,
            'total_outputs': len(result.outputs),
            'error_message': result.error_message
        }
        
        if result.success:
            log_data.update({
                'successful_outputs': len([o for o in result.outputs if not (o.metadata and o.metadata.get('error'))]),
                'output_types': list(set([o.output_type for o in result.outputs])),
                'generators_used': list(set([
                    o.metadata.get('generator_type', 'unknown') 
                    for o in result.outputs 
                    if o.metadata and not o.metadata.get('error')
                ]))
            })
        
        log_file = self.output_directory / "processing_log.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write processing log: {e}")
    
    async def process(self, input_data: Any) -> Any:
        """Implementation of PipelineComponent interface"""
        if isinstance(input_data, tuple) and len(input_data) == 2:
            outputs, original_input = input_data
            return await self.integrate_outputs(outputs, original_input)
        else:
            raise ValueError("OutputIntegrator expects tuple of (outputs, original_input)")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        
        session_dir = self.output_directory / session_id
        
        if not session_dir.exists():
            return None
        
        files = list(session_dir.iterdir())
        
        return {
            'session_id': session_id,
            'directory': str(session_dir),
            'file_count': len(files),
            'files': [f.name for f in files],
            'total_size_bytes': sum(f.stat().st_size for f in files if f.is_file()),
            'created_at': datetime.fromtimestamp(session_dir.stat().st_ctime).isoformat()
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all processing sessions"""
        
        sessions = []
        
        for item in self.output_directory.iterdir():
            if item.is_dir() and item.name.startswith('wb_session_'):
                session_info = self.get_session_info(item.name)
                if session_info:
                    sessions.append(session_info)
        
        return sorted(sessions, key=lambda s: s['created_at'], reverse=True)