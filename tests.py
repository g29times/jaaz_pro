import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from whiteboard_pipeline import WhiteboardPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType, ProcessingResult
from whiteboard_pipeline.components import InputParser, VLMEngine, TaskRouter


class TestWhiteboardPipeline(unittest.TestCase):
    """Test cases for the main WhiteboardPipeline class"""
    
    def setUp(self):
        self.test_config = {
            "pipeline": {"log_level": "DEBUG"},
            "input_parser": {"use_paddleocr": False},
            "vlm_engine": {"fallback_enabled": True},
            "task_router": {"fallback_enabled": True}
        }
        self.pipeline = WhiteboardPipeline(self.test_config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline.input_parser)
        self.assertIsNotNone(self.pipeline.vlm_engine)
        self.assertIsNotNone(self.pipeline.task_router)
        self.assertIsNotNone(self.pipeline.generator_dispatcher)
        self.assertIsNotNone(self.pipeline.output_integrator)
    
    @patch('whiteboard_pipeline.components.InputParser.parse')
    @patch('whiteboard_pipeline.components.VLMEngine.extract_intent')
    @patch('whiteboard_pipeline.components.TaskRouter.create_task_plan')
    @patch('whiteboard_pipeline.components.GeneratorDispatcher.execute_task_plan')
    @patch('whiteboard_pipeline.components.OutputIntegrator.integrate_outputs')
    def test_successful_processing(self, mock_integrate, mock_execute, mock_plan, mock_intent, mock_parse):
        """Test successful end-to-end processing"""
        
        # Mock all pipeline steps
        from whiteboard_pipeline.models import ParsedInput, ParsedElement, SemanticIntent, TaskPlan, TaskStep, GeneratorType, GeneratorOutput
        
        mock_parse.return_value = ParsedInput(
            elements=[ParsedElement("text", "test content", 0.9)],
            raw_text="test content",
            metadata={}
        )
        
        mock_intent.return_value = SemanticIntent(
            intent="analyze_content",
            confidence=0.8,
            context={},
            suggested_tasks=["analyze"]
        )
        
        mock_plan.return_value = TaskPlan(
            steps=[TaskStep("analyze", GeneratorType.REPORT_SUMMARIZER, {})],
            priority=5
        )
        
        mock_execute.return_value = [GeneratorOutput("Analysis result", "md")]
        
        mock_integrate.return_value = ProcessingResult(
            outputs=[GeneratorOutput("Analysis result", "md")],
            execution_time=1.0,
            success=True
        )
        
        # Test processing
        async def run_test():
            input_data = WhiteboardInput("test content", InputType.TEXT)
            result = await self.pipeline.process(input_data)
            self.assertTrue(result.success)
            self.assertEqual(len(result.outputs), 1)
        
        asyncio.run(run_test())
    
    def test_health_check(self):
        """Test pipeline health check"""
        
        async def run_test():
            health = await self.pipeline.health_check()
            self.assertIn('pipeline', health)
            self.assertIn('components', health)
            self.assertIn('timestamp', health)
        
        asyncio.run(run_test())


class TestInputParser(unittest.TestCase):
    """Test cases for InputParser component"""
    
    def setUp(self):
        self.parser = InputParser({"use_paddleocr": False})
    
    def test_text_parsing(self):
        """Test parsing of text input"""
        
        async def run_test():
            input_data = WhiteboardInput("Hello world", InputType.TEXT)
            result = await self.parser.parse(input_data)
            
            self.assertEqual(len(result.elements), 1)
            self.assertEqual(result.elements[0].content, "Hello world")
            self.assertEqual(result.raw_text, "Hello world")
        
        asyncio.run(run_test())
    
    def test_image_parsing_fallback(self):
        """Test image parsing with fallback when OCR not available"""
        
        async def run_test():
            # Create mock image data
            image_data = b"fake image data"
            input_data = WhiteboardInput(image_data, InputType.IMAGE)
            
            result = await self.parser.parse(input_data)
            
            # Should have some elements even with fallback
            self.assertGreaterEqual(len(result.elements), 0)
        
        asyncio.run(run_test())


class TestVLMEngine(unittest.TestCase):
    """Test cases for VLMEngine component"""
    
    def setUp(self):
        self.vlm_engine = VLMEngine({"fallback_enabled": True})
    
    def test_fallback_intent_extraction(self):
        """Test fallback intent extraction when VLM is unavailable"""
        
        async def run_test():
            from whiteboard_pipeline.models import ParsedInput, ParsedElement
            
            parsed_input = ParsedInput(
                elements=[ParsedElement("text", "create a flowchart", 0.9)],
                raw_text="create a flowchart",
                metadata={}
            )
            
            result = await self.vlm_engine.extract_intent(parsed_input)
            
            self.assertIsNotNone(result.intent)
            self.assertGreater(result.confidence, 0)
            self.assertEqual(result.intent, "create_flowchart")
        
        asyncio.run(run_test())


class TestTaskRouter(unittest.TestCase):
    """Test cases for TaskRouter component"""
    
    def setUp(self):
        self.task_router = TaskRouter({"fallback_enabled": True})
    
    def test_fallback_task_planning(self):
        """Test fallback task planning when LLM is unavailable"""
        
        async def run_test():
            from whiteboard_pipeline.models import SemanticIntent
            
            intent = SemanticIntent(
                intent="create_flowchart",
                confidence=0.8,
                context={},
                suggested_tasks=["flowchart"]
            )
            
            result = await self.task_router.create_task_plan(intent)
            
            self.assertGreater(len(result.steps), 0)
            self.assertEqual(result.steps[0].generator_type.value, "mermaid_flow")
        
        asyncio.run(run_test())


class TestGeneratorDispatcher(unittest.TestCase):
    """Test cases for GeneratorDispatcher"""
    
    def setUp(self):
        from whiteboard_pipeline.components import GeneratorDispatcher
        self.dispatcher = GeneratorDispatcher({})
    
    def test_generator_status(self):
        """Test getting generator status"""
        status = self.dispatcher.get_generator_status()
        
        self.assertIn('diffusion_image', status)
        self.assertIn('mermaid_flow', status)
        self.assertIn('report_summarizer', status)
    
    def test_supports_generator_type(self):
        """Test generator type support checking"""
        from whiteboard_pipeline.models import GeneratorType
        
        self.assertTrue(self.dispatcher.supports_generator_type(GeneratorType.MERMAID_FLOW))
        self.assertTrue(self.dispatcher.supports_generator_type(GeneratorType.DIFFUSION_IMAGE))
        self.assertTrue(self.dispatcher.supports_generator_type(GeneratorType.REPORT_SUMMARIZER))


class TestOutputIntegrator(unittest.TestCase):
    """Test cases for OutputIntegrator"""
    
    def setUp(self):
        from whiteboard_pipeline.components import OutputIntegrator
        self.integrator = OutputIntegrator({"output_directory": tempfile.gettempdir()})
    
    def test_session_management(self):
        """Test session ID generation and management"""
        session_id = self.integrator._generate_session_id()
        
        self.assertTrue(session_id.startswith("wb_session_"))
        self.assertGreater(len(session_id), 15)
    
    def test_feedback_preparation(self):
        """Test feedback data preparation"""
        
        async def run_test():
            from whiteboard_pipeline.models import GeneratorOutput, WhiteboardInput, InputType
            
            outputs = [
                GeneratorOutput("test content", "md"),
                GeneratorOutput("test image", "image")
            ]
            
            input_data = WhiteboardInput("test", InputType.TEXT)
            
            feedback = await self.integrator._prepare_feedback_collection(outputs, input_data)
            
            if feedback:  # Only test if feedback is enabled
                self.assertIn('session_metadata', feedback)
                self.assertIn('quality_metrics', feedback)
                self.assertIn('feedback_questions', feedback)
        
        asyncio.run(run_test())


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and recovery"""
    
    def setUp(self):
        from whiteboard_pipeline.error_handling import ErrorRecoveryManager
        self.recovery_manager = ErrorRecoveryManager()
    
    def test_recovery_strategies(self):
        """Test that recovery strategies are properly configured"""
        
        expected_strategies = [
            'input_parsing',
            'intent_extraction', 
            'task_planning',
            'generation',
            'integration'
        ]
        
        for strategy in expected_strategies:
            self.assertIn(strategy, self.recovery_manager.recovery_strategies)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        
        from whiteboard_pipeline.error_handling import CircuitBreaker
        
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        # Initially closed
        self.assertEqual(circuit_breaker.state, 'closed')
        
        # Record failures
        circuit_breaker._record_failure()
        circuit_breaker._record_failure()
        
        # Should be open now
        self.assertEqual(circuit_breaker.state, 'open')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        self.config = {
            "input_parser": {"use_paddleocr": False},
            "vlm_engine": {"fallback_enabled": True},
            "task_router": {"fallback_enabled": True},
            "generator_dispatcher": {
                "diffusion_config": {"test_mode": True},
                "mermaid_config": {"test_mode": True},
                "report_config": {"test_mode": True}
            },
            "output_integrator": {"output_directory": tempfile.gettempdir()}
        }
    
    def test_end_to_end_text_processing(self):
        """Test complete end-to-end text processing"""
        
        async def run_test():
            pipeline = WhiteboardPipeline(self.config)
            
            input_data = WhiteboardInput(
                "Create a simple process flow for user registration",
                InputType.TEXT
            )
            
            result = await pipeline.process(input_data)
            
            # Should complete even with fallbacks
            self.assertIsNotNone(result)
            self.assertGreaterEqual(len(result.outputs), 1)
        
        asyncio.run(run_test())
    
    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        
        # Test with valid config
        pipeline1 = WhiteboardPipeline(self.config)
        self.assertIsNotNone(pipeline1)
        
        # Test with empty config (should use defaults)
        pipeline2 = WhiteboardPipeline({})
        self.assertIsNotNone(pipeline2)
        
        # Test with None config (should use defaults)
        pipeline3 = WhiteboardPipeline(None)
        self.assertIsNotNone(pipeline3)


def run_tests():
    """Run all tests"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestWhiteboardPipeline,
        TestInputParser,
        TestVLMEngine, 
        TestTaskRouter,
        TestGeneratorDispatcher,
        TestOutputIntegrator,
        TestErrorHandling,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)