import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType
from whiteboard_pipeline.components import InputParser, VLMEngine, MermaidFlowGenerator


class TestSimpleSketchToMermaidPipeline:
    """Test the simplified Sketch → Mermaid pipeline"""
    
    @pytest.fixture
    def pipeline_config(self):
        return {
            "pipeline": {"log_level": "DEBUG"},
            "input_parser": {"ocr_confidence_threshold": 0.3},
            "vlm_engine": {"fallback_enabled": True},
            "mermaid_generator": {"fallback_enabled": True}
        }
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        return SimpleSketchToMermaidPipeline(pipeline_config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly with core components"""
        assert pipeline.input_parser is not None
        assert pipeline.vlm_engine is not None
        assert pipeline.mermaid_generator is not None
        assert isinstance(pipeline.session_logs, list)
    
    @pytest.mark.asyncio
    async def test_text_to_mermaid_workflow(self, pipeline):
        """Test complete text → Mermaid workflow"""
        
        # Mock the components to avoid external dependencies
        with patch.object(pipeline.input_parser, 'parse') as mock_parse, \
             patch.object(pipeline.vlm_engine, 'extract_intent') as mock_intent, \
             patch.object(pipeline.mermaid_generator, 'generate') as mock_generate:
            
            # Setup mocks
            from whiteboard_pipeline.models import ParsedInput, ParsedElement, SemanticIntent, GeneratorOutput
            
            mock_parse.return_value = ParsedInput(
                elements=[ParsedElement("text", "user login process", 0.9)],
                raw_text="user login process: 1. enter credentials 2. validate 3. grant access",
                metadata={"processing_method": "direct_text"}
            )
            
            mock_intent.return_value = SemanticIntent(
                intent="CREATE_FLOWCHART",
                confidence=0.8,
                context={"domain": "process", "flow_direction": "TD"},
                suggested_tasks=["create_mermaid_flowchart"]
            )
            
            mock_generate.return_value = GeneratorOutput(
                content="flowchart TD\n    A([Start]) --> B[Enter Credentials]\n    B --> C[Validate]\n    C --> D([End])",
                output_type="mermaid",
                file_path=Path("/tmp/test_flowchart.mmd"),
                metadata={"success": True}
            )
            
            # Test the workflow
            input_data = WhiteboardInput("user login process", InputType.TEXT)
            result = await pipeline.process_sketch_to_mermaid(input_data)
            
            # Verify results
            assert result.success is True
            assert len(result.outputs) == 1
            assert result.outputs[0].output_type == "mermaid"
            assert "flowchart TD" in result.outputs[0].content
            
            # Verify all components were called
            mock_parse.assert_called_once()
            mock_intent.assert_called_once()
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_logging(self, pipeline):
        """Test error handling creates proper feedback logs"""
        
        with patch.object(pipeline.input_parser, 'parse') as mock_parse:
            mock_parse.side_effect = Exception("OCR failed")
            
            input_data = WhiteboardInput("test input", InputType.TEXT)
            result = await pipeline.process_sketch_to_mermaid(input_data)
            
            # Verify error handling
            assert result.success is False
            assert "OCR failed" in result.error_message
            assert result.feedback_data is not None
            assert result.feedback_data['session_log']['success'] is False
            
            # Verify session was logged
            assert len(pipeline.session_logs) == 1
            assert pipeline.session_logs[0]['success'] is False
    
    @pytest.mark.asyncio 
    async def test_health_check(self, pipeline):
        """Test health check functionality"""
        
        # Mock OCR engines to simulate healthy state
        pipeline.input_parser.ocr_engine = Mock()
        pipeline.vlm_engine.engine = Mock()
        
        health = await pipeline.health_check()
        
        assert 'pipeline' in health
        assert 'components' in health
        assert 'focus' in health
        assert health['focus'] == 'sketch_to_mermaid'
        assert health['components']['input_parser'] == 'healthy'
    
    def test_session_analytics(self, pipeline):
        """Test session analytics and feedback collection"""
        
        # Add some mock session data
        pipeline.session_logs = [
            {
                'session_id': 'test1',
                'success': True,
                'total_duration': 2.5,
                'steps': [
                    {'step': 'input_parsing', 'duration': 0.5, 'success': True},
                    {'step': 'intent_extraction', 'duration': 1.0, 'success': True},
                    {'step': 'mermaid_generation', 'duration': 1.0, 'success': True}
                ]
            },
            {
                'session_id': 'test2', 
                'success': False,
                'total_duration': 1.0,
                'error': 'Test error',
                'steps': [
                    {'step': 'input_parsing', 'duration': 1.0, 'success': False}
                ]
            }
        ]
        
        analytics = pipeline.get_session_analytics()
        
        assert analytics['total_sessions'] == 2
        assert analytics['successful_sessions'] == 1
        assert analytics['failed_sessions'] == 1
        assert analytics['success_rate'] == 0.5
        assert 'step_performance' in analytics
        assert analytics['focus_workflow'] == 'sketch_to_mermaid'


class TestInputParser:
    """Test mandatory OCR functionality"""
    
    @pytest.fixture
    def parser_config(self):
        return {"ocr_confidence_threshold": 0.3}
    
    def test_initialization_requires_ocr(self, parser_config):
        """Test that InputParser requires at least one OCR engine"""
        
        with patch('whiteboard_pipeline.components.input_parser.PaddleOCR', side_effect=ImportError), \
             patch('whiteboard_pipeline.components.input_parser.easyocr', side_effect=ImportError):
            
            with pytest.raises(RuntimeError, match="No OCR engines available"):
                InputParser(parser_config)
    
    @pytest.mark.asyncio
    async def test_mandatory_ocr_text_processing(self):
        """Test that text processing includes flowchart keyword detection"""
        
        parser = InputParser({"ocr_confidence_threshold": 0.3})
        
        # Mock OCR engines to avoid dependency
        parser.ocr_engine = Mock()
        parser.backup_ocr_engine = Mock()
        
        input_data = WhiteboardInput(
            "Process flow: 1. Start 2. Decision point 3. End",
            InputType.TEXT
        )
        
        result = await parser.parse(input_data)
        
        assert len(result.elements) == 1
        assert result.elements[0].element_type == "text"
        assert "flowchart_keywords" in result.elements[0].metadata
        
        # Should detect flowchart keywords
        keywords = result.elements[0].metadata["flowchart_keywords"]
        assert "process" in keywords
        assert "flow" in keywords
    
    @pytest.mark.asyncio
    async def test_mandatory_ocr_failure(self):
        """Test that OCR failure raises appropriate error"""
        
        parser = InputParser({"ocr_confidence_threshold": 0.3})
        
        # Mock both OCR engines to fail
        parser.ocr_engine = Mock()
        parser.backup_ocr_engine = Mock()
        
        with patch.object(parser, '_ocr_with_paddleocr', side_effect=Exception("OCR failed")), \
             patch.object(parser, '_ocr_with_easyocr', side_effect=Exception("Backup OCR failed")):
            
            input_data = WhiteboardInput(b"fake image data", InputType.IMAGE)
            
            with pytest.raises(RuntimeError, match="Mandatory OCR failed"):
                await parser.parse(input_data)


class TestVLMEngine:
    """Test vLLM integration and fallback"""
    
    @pytest.fixture
    def vlm_config(self):
        return {
            "model_name": "Qwen/Qwen-VL-Chat",
            "tensor_parallel_size": 1,
            "temperature": 0.1
        }
    
    def test_initialization_without_vllm(self, vlm_config):
        """Test VLM engine gracefully handles missing vLLM"""
        
        with patch('whiteboard_pipeline.components.vlm_engine.VLLM_AVAILABLE', False):
            vlm = VLMEngine(vlm_config)
            assert vlm.engine is None
    
    @pytest.mark.asyncio
    async def test_fallback_intent_extraction(self, vlm_config):
        """Test fallback intent extraction focuses on flowchart creation"""
        
        vlm = VLMEngine(vlm_config)
        vlm.engine = None  # Force fallback mode
        
        from whiteboard_pipeline.models import ParsedInput, ParsedElement
        
        parsed_input = ParsedInput(
            elements=[
                ParsedElement("text", "process workflow", 0.9),
                ParsedElement("arrow", "-", 0.7)
            ],
            raw_text="create a process workflow with decision points",
            metadata={}
        )
        
        result = await vlm.extract_intent(parsed_input)
        
        # Should default to CREATE_FLOWCHART with good confidence
        assert result.intent == "CREATE_FLOWCHART"
        assert result.confidence > 0.6  # Should have decent confidence
        assert result.context["domain"] == "process"
        assert result.context["has_process_steps"] is True
        assert "create_mermaid_flowchart" in result.suggested_tasks


class TestMermaidFlowGenerator:
    """Test Mermaid generation with comprehensive logging"""
    
    @pytest.fixture
    def generator_config(self):
        return {
            "llm_provider": "openai",
            "api_key": "test-key",
            "model_name": "gpt-4"
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        return MermaidFlowGenerator(generator_config)
    
    def test_generation_statistics_tracking(self, generator):
        """Test that generation statistics are properly tracked"""
        
        initial_stats = generator.get_generation_stats()
        
        assert initial_stats['total_generations'] == 0
        assert initial_stats['success_rate'] == 0.0
        assert initial_stats['llm_usage_rate'] == 0.0
        assert initial_stats['fallback_usage_rate'] == 0.0
    
    @pytest.mark.asyncio
    async def test_fallback_mermaid_generation(self, generator):
        """Test structured fallback Mermaid generation"""
        
        from whiteboard_pipeline.models import TaskStep, GeneratorType
        
        task_step = TaskStep(
            action="create_mermaid_flowchart",
            generator_type=GeneratorType.MERMAID_FLOW,
            parameters={
                "content": "1. Login user 2. Validate credentials 3. Grant access",
                "direction": "TD"
            }
        )
        
        context = {"session_id": "test_session"}
        
        # Mock API call to force fallback
        generator.api_key = None
        
        result = await generator.generate(task_step, context)
        
        assert result.output_type == "mermaid"
        assert "flowchart TD" in result.content
        assert result.metadata["success"] is True
        assert result.metadata["generation_method"] == "fallback"
        
        # Check statistics update
        stats = generator.get_generation_stats()
        assert stats['total_generations'] == 1
        assert stats['successful_generations'] == 1
        assert stats['fallback_generations'] == 1
    
    def test_process_step_extraction(self, generator):
        """Test extraction of process steps from content"""
        
        content = """
        User Registration Process:
        1. User enters email and password
        2. System validates email format
        3. Check if user already exists
        4. Create new user account
        5. Send confirmation email
        """
        
        steps = generator._extract_process_steps(content, "test_session")
        
        assert len(steps) > 0
        assert any("email" in step.lower() for step in steps)
        assert any("validate" in step.lower() for step in steps)
    
    def test_mermaid_syntax_validation(self, generator):
        """Test Mermaid syntax validation"""
        
        # Valid Mermaid code
        valid_code = """flowchart TD
    A([Start]) --> B[Process]
    B --> C([End])"""
        
        assert generator._validate_mermaid_syntax(valid_code, "test_session") is True
        
        # Invalid Mermaid code
        invalid_code = "This is not Mermaid code"
        assert generator._validate_mermaid_syntax(invalid_code, "test_session") is False
        
        # Missing arrows
        no_arrows = "flowchart TD\n    A[Start]\n    B[End]"
        assert generator._validate_mermaid_syntax(no_arrows, "test_session") is False


class TestIntegration:
    """Integration tests for the simplified pipeline"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_sketch_to_mermaid(self):
        """Test complete end-to-end sketch to Mermaid conversion"""
        
        config = {
            "input_parser": {"ocr_confidence_threshold": 0.3},
            "vlm_engine": {"fallback_enabled": True},
            "mermaid_generator": {"fallback_enabled": True}
        }
        
        pipeline = SimpleSketchToMermaidPipeline(config)
        
        # Mock components to avoid external dependencies
        pipeline.input_parser.ocr_engine = Mock()
        pipeline.input_parser.backup_ocr_engine = Mock()
        pipeline.vlm_engine.engine = None  # Force fallback
        pipeline.mermaid_generator.api_key = None  # Force fallback
        
        input_data = WhiteboardInput(
            "Create a simple login workflow with validation",
            InputType.TEXT
        )
        
        result = await pipeline.process_sketch_to_mermaid(input_data)
        
        # Should succeed even with all fallbacks
        assert result.success is True
        assert len(result.outputs) == 1
        assert result.outputs[0].output_type == "mermaid"
        assert "flowchart" in result.outputs[0].content
        
        # Should have comprehensive feedback data
        assert result.feedback_data is not None
        assert 'session_log' in result.feedback_data
        assert result.feedback_data['pipeline_type'] == 'sketch_to_mermaid'
        
        # Check that analytics work
        analytics = pipeline.get_session_analytics()
        assert analytics['total_sessions'] == 1
        assert analytics['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_logging_and_feedback_collection(self):
        """Test comprehensive logging for feedback collection"""
        
        config = {"log_file": "test_feedback.log"}
        pipeline = SimpleSketchToMermaidPipeline(config)
        
        # Mock components
        pipeline.input_parser.ocr_engine = Mock()
        pipeline.vlm_engine.engine = None
        pipeline.mermaid_generator.api_key = None
        
        # Process multiple inputs to generate logs
        inputs = [
            WhiteboardInput("Process A", InputType.TEXT),
            WhiteboardInput("Process B", InputType.TEXT)
        ]
        
        for input_data in inputs:
            await pipeline.process_sketch_to_mermaid(input_data)
        
        # Verify session logging
        assert len(pipeline.session_logs) == 2
        
        analytics = pipeline.get_session_analytics()
        assert analytics['total_sessions'] == 2
        assert 'step_performance' in analytics
        assert 'recent_errors' in analytics


def test_pytest_configuration():
    """Test that pytest is properly configured"""
    
    # This test ensures pytest is working with async support
    assert pytest.__version__ >= "7.0.0"
    
    # Test async support
    async def dummy_async():
        return True
    
    # Should be able to run async code in tests
    result = asyncio.run(dummy_async())
    assert result is True


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])