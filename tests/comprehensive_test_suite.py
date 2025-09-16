"""
Comprehensive Test Suite with Visualization
Creates organized tests with visual output for Phase 2 capabilities
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import tempfile
import base64
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Image processing imports
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pipeline imports
from whiteboard_pipeline.simple_pipeline import SimpleSketchToMermaidPipeline
from whiteboard_pipeline.models import WhiteboardInput, InputType
from whiteboard_pipeline.components.image_processor import ImageProcessor


class TestVisualizer:
    """Visualize test results with annotated images and reports"""
    
    def __init__(self, test_folder: Path):
        self.test_folder = test_folder
        self.outputs_folder = test_folder / "outputs"
        self.outputs_folder.mkdir(exist_ok=True)
        
        # Create subfolders for different types of outputs
        (self.outputs_folder / "annotated_images").mkdir(exist_ok=True)
        (self.outputs_folder / "mermaid_files").mkdir(exist_ok=True)
        (self.outputs_folder / "analysis_reports").mkdir(exist_ok=True)
        (self.outputs_folder / "visual_comparisons").mkdir(exist_ok=True)
        
    def visualize_detection_results(self, image_path: Path, elements: List, output_name: str):
        """Create annotated image showing detected elements"""
        try:
            # Load original image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Could not load image: {image_path}")
                return None
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = img_rgb.copy()
            
            # Color scheme for different element types
            colors = {
                'shape': (255, 0, 0),      # Red
                'arrow': (0, 255, 0),      # Green  
                'text_region': (0, 0, 255), # Blue
                'text': (255, 255, 0),      # Yellow
                'polygon': (255, 0, 255),   # Magenta
                'rectangle': (0, 255, 255), # Cyan
            }
            
            element_stats = {}
            
            for element in elements:
                element_type = element.element_type
                element_stats[element_type] = element_stats.get(element_type, 0) + 1
                
                # Get bounding box if available
                if hasattr(element, 'metadata') and element.metadata and 'bounding_box' in element.metadata:
                    bbox = element.metadata['bounding_box']
                    
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        color = colors.get(element_type, (128, 128, 128))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                        
                        # Add label
                        label = f"{element_type} ({element.confidence:.2f})"
                        cv2.putText(annotated, label, (int(x), int(y - 5)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save annotated image
            output_path = self.outputs_folder / "annotated_images" / f"{output_name}_annotated.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            
            # Create statistics overlay
            self._create_stats_overlay(img_rgb, element_stats, output_name)
            
            return output_path, element_stats
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None, {}
    
    def _create_stats_overlay(self, original_img: np.ndarray, stats: Dict, output_name: str):
        """Create an image with statistics overlay"""
        try:
            # Create a side-by-side layout
            height, width = original_img.shape[:2]
            stats_width = 400
            combined = np.ones((height, width + stats_width, 3), dtype=np.uint8) * 255
            
            # Place original image
            combined[:height, :width] = original_img
            
            # Add statistics text
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 0, 0)
            thickness = 1
            
            # Title
            cv2.putText(combined, "Detection Results", (width + 10, y_offset), 
                       font, 0.8, color, 2)
            y_offset += 40
            
            # Statistics
            total_elements = sum(stats.values())
            cv2.putText(combined, f"Total Elements: {total_elements}", 
                       (width + 10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
            
            for element_type, count in sorted(stats.items()):
                text = f"{element_type}: {count}"
                cv2.putText(combined, text, (width + 10, y_offset), 
                           font, font_scale, color, thickness)
                y_offset += 20
            
            # Save combined image
            output_path = self.outputs_folder / "visual_comparisons" / f"{output_name}_with_stats.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error creating stats overlay: {e}")
    
    def create_analysis_report(self, test_name: str, results: Dict[str, Any]):
        """Create detailed analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.outputs_folder / "analysis_reports" / f"{test_name}_{timestamp}_report.json"
        
        # Enhanced report with metadata
        report = {
            "test_name": test_name,
            "timestamp": timestamp,
            "results": results,
            "summary": {
                "success": results.get("success", False),
                "processing_time": results.get("processing_time", 0),
                "elements_detected": len(results.get("elements", [])),
                "pipeline_steps": len(results.get("pipeline_steps", [])),
            }
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable report
        txt_report_path = self.outputs_folder / "analysis_reports" / f"{test_name}_{timestamp}_report.txt"
        with open(txt_report_path, 'w') as f:
            f.write(f"Test Analysis Report: {test_name}\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Success: {'✅' if results.get('success') else '❌'}\n")
            f.write(f"Processing Time: {results.get('processing_time', 0):.3f}s\n")
            f.write(f"Elements Detected: {len(results.get('elements', []))}\n\n")
            
            # Element breakdown
            if 'element_stats' in results:
                f.write("Element Breakdown:\n")
                for element_type, count in results['element_stats'].items():
                    f.write(f"  • {element_type}: {count}\n")
                f.write("\n")
            
            # Generated content
            if 'mermaid_content' in results:
                f.write("Generated Mermaid:\n")
                f.write(results['mermaid_content'])
                f.write("\n\n")
            
            # Pipeline performance
            if 'pipeline_steps' in results:
                f.write("Pipeline Performance:\n")
                for step in results['pipeline_steps']:
                    f.write(f"  • {step['step']}: {step['duration']:.4f}s\n")
        
        return report_path, txt_report_path


class ComprehensiveTestSuite:
    """Comprehensive test suite with visualization"""
    
    def __init__(self, test_folder: Path):
        self.test_folder = test_folder
        self.images_folder = test_folder / "images"
        self.visualizer = TestVisualizer(test_folder)
        
        # Set environment variable
        if not os.environ.get('OPENAI_API_KEY'):
            os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
    
    async def run_comprehensive_tests(self):
        """Run all tests with visualization"""
        print("🧪 Comprehensive Test Suite with Visualization")
        print("=" * 60)
        
        # Find all test images
        test_images = list(self.images_folder.glob("*.png")) + list(self.images_folder.glob("*.jpg"))
        
        if not test_images:
            print("No test images found. Creating sample images...")
            test_images = await self._create_sample_test_images()
        
        print(f"Found {len(test_images)} test images:")
        for img in test_images:
            print(f"  📄 {img.name}")
        
        # Test results storage
        all_results = {}
        
        # Test each image
        for image_path in test_images:
            print(f"\n{'='*50}")
            print(f"🔍 Testing: {image_path.name}")
            print(f"{'='*50}")
            
            test_name = image_path.stem
            results = await self._test_single_image(image_path, test_name)
            all_results[test_name] = results
        
        # Create comprehensive summary
        await self._create_test_summary(all_results)
        
        print(f"\n{'='*60}")
        print(f"✅ Comprehensive testing completed!")
        print(f"📁 Results saved in: {self.test_folder / 'outputs'}")
        print(f"🎨 Visualizations available in subfolders")
        
        return all_results
    
    async def _test_single_image(self, image_path: Path, test_name: str) -> Dict[str, Any]:
        """Test a single image with full analysis and visualization"""
        results = {
            "image_path": str(image_path),
            "test_name": test_name,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Phase 1: Direct ImageProcessor analysis
            print("🔍 Phase 1: Computer Vision Analysis")
            processor = ImageProcessor()
            
            input_data = WhiteboardInput(
                content=image_path,
                input_type=InputType.IMAGE,
                metadata={"test": test_name, "source": "test_suite"}
            )
            
            cv_result = await processor.process_image(input_data)
            
            # Store computer vision results
            results["elements"] = [
                {
                    "type": e.element_type,
                    "content": e.content[:100],  # Truncate for readability
                    "confidence": e.confidence,
                    "metadata": e.metadata
                }
                for e in cv_result.elements
            ]
            
            # Create element statistics
            element_stats = {}
            for element in cv_result.elements:
                element_type = element.element_type
                element_stats[element_type] = element_stats.get(element_type, 0) + 1
            
            results["element_stats"] = element_stats
            
            print(f"   ✅ Detected {len(cv_result.elements)} elements")
            print(f"   📊 Element breakdown: {element_stats}")
            
            # Phase 2: Create visualizations
            print("🎨 Phase 2: Creating Visualizations")
            viz_path, viz_stats = self.visualizer.visualize_detection_results(
                image_path, cv_result.elements, test_name
            )
            
            if viz_path:
                print(f"   ✅ Visualization saved: {viz_path.name}")
                results["visualization_path"] = str(viz_path)
            
            # Phase 3: Full pipeline test
            print("🚀 Phase 3: Full Pipeline Processing")
            
            config = {
                "pipeline": {"log_level": "INFO"},
                "input_parser": {
                    "ocr_confidence_threshold": 0.3,
                    "image_processor": {
                        "min_contour_area": 100,
                        "max_contour_area": 50000
                    }
                },
                "vlm_engine": {"fallback_enabled": True},
                "mermaid_generator": {"fallback_enabled": True}
            }
            
            pipeline = SimpleSketchToMermaidPipeline(config)
            start_time = datetime.now()
            
            pipeline_result = await pipeline.process_sketch_to_mermaid(input_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time
            results["success"] = pipeline_result.success
            
            if pipeline_result.success:
                print(f"   ✅ Pipeline successful in {processing_time:.3f}s")
                
                # Save Mermaid output
                mermaid_output = pipeline_result.outputs[0]
                mermaid_path = self.visualizer.outputs_folder / "mermaid_files" / f"{test_name}_output.mmd"
                
                with open(mermaid_path, 'w') as f:
                    f.write(mermaid_output.content)
                
                results["mermaid_path"] = str(mermaid_path)
                results["mermaid_content"] = mermaid_output.content
                
                # Pipeline performance data
                if pipeline_result.feedback_data and 'session_log' in pipeline_result.feedback_data:
                    session_log = pipeline_result.feedback_data['session_log']
                    results["pipeline_steps"] = session_log.get('steps', [])
                
                print(f"   📄 Mermaid saved: {mermaid_path.name}")
                
            else:
                print(f"   ❌ Pipeline failed: {pipeline_result.error_message}")
                results["error"] = pipeline_result.error_message
            
            # Phase 4: Generate report
            print("📝 Phase 4: Generating Analysis Report")
            report_json, report_txt = self.visualizer.create_analysis_report(test_name, results)
            
            results["report_json"] = str(report_json)
            results["report_txt"] = str(report_txt)
            
            print(f"   ✅ Reports saved: {report_json.name}, {report_txt.name}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results["error"] = str(e)
            results["success"] = False
        
        return results
    
    async def _create_sample_test_images(self) -> List[Path]:
        """Create sample test images if none exist"""
        sample_images = []
        
        # Create a simple flowchart
        img1 = Image.new('RGB', (600, 400), color='white')
        draw1 = ImageDraw.Draw(img1)
        
        # Start oval
        draw1.ellipse([250, 50, 350, 100], outline='black', width=2)
        draw1.text((285, 70), "START", fill='black')
        
        # Process box
        draw1.rectangle([250, 150, 350, 200], outline='black', width=2)
        draw1.text((270, 170), "PROCESS", fill='black')
        
        # Decision diamond (approximated with polygon)
        draw1.polygon([(300, 250), (350, 275), (300, 300), (250, 275)], outline='black', width=2)
        draw1.text((290, 270), "?", fill='black')
        
        # End oval
        draw1.ellipse([250, 350, 350, 400], outline='black', width=2)
        draw1.text((290, 370), "END", fill='black')
        
        # Arrows
        draw1.line([300, 100, 300, 150], fill='black', width=2)
        draw1.line([300, 200, 300, 250], fill='black', width=2)
        draw1.line([300, 300, 300, 350], fill='black', width=2)
        
        path1 = self.images_folder / "sample_flowchart.png"
        img1.save(path1)
        sample_images.append(path1)
        
        # Create a text-heavy image
        img2 = Image.new('RGB', (500, 300), color='white')
        draw2 = ImageDraw.Draw(img2)
        
        draw2.text((50, 50), "Project Workflow:", fill='black')
        draw2.text((50, 100), "1. Planning Phase", fill='black')
        draw2.text((50, 130), "2. Development Phase", fill='black')
        draw2.text((50, 160), "3. Testing Phase", fill='black')
        draw2.text((50, 190), "4. Deployment Phase", fill='black')
        draw2.text((50, 220), "5. Maintenance Phase", fill='black')
        
        path2 = self.images_folder / "sample_text_workflow.png"
        img2.save(path2)
        sample_images.append(path2)
        
        print(f"Created {len(sample_images)} sample test images")
        return sample_images
    
    async def _create_test_summary(self, all_results: Dict[str, Any]):
        """Create comprehensive test summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.visualizer.outputs_folder / f"test_summary_{timestamp}.html"
        
        # Generate HTML summary
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 2 Test Results Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .test-case {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 8px; }}
        .success {{ border-left: 5px solid #4CAF50; }}
        .failure {{ border-left: 5px solid #f44336; }}
        .stats {{ display: flex; gap: 20px; margin: 10px 0; }}
        .stat-box {{ background: #f9f9f9; padding: 10px; border-radius: 4px; min-width: 120px; }}
        .mermaid-code {{ background: #f5f5f5; padding: 15px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Phase 2 Test Results Summary</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Tests:</strong> {len(all_results)}</p>
        <p><strong>Success Rate:</strong> {sum(1 for r in all_results.values() if r.get('success', False)) / len(all_results) * 100:.1f}%</p>
    </div>
"""
        
        for test_name, results in all_results.items():
            success_class = "success" if results.get('success', False) else "failure"
            status_icon = "✅" if results.get('success', False) else "❌"
            
            html_content += f"""
    <div class="test-case {success_class}">
        <h2>{status_icon} {test_name}</h2>
        
        <div class="stats">
            <div class="stat-box">
                <strong>Processing Time</strong><br>
                {results.get('processing_time', 0):.3f}s
            </div>
            <div class="stat-box">
                <strong>Elements Detected</strong><br>
                {len(results.get('elements', []))}
            </div>
            <div class="stat-box">
                <strong>Pipeline Steps</strong><br>
                {len(results.get('pipeline_steps', []))}
            </div>
        </div>
        
        <h3>Element Breakdown:</h3>
        <ul>
"""
            
            for element_type, count in results.get('element_stats', {}).items():
                html_content += f"<li><strong>{element_type}:</strong> {count}</li>"
            
            html_content += "</ul>"
            
            # Add visualization if available
            if 'visualization_path' in results:
                rel_path = Path(results['visualization_path']).name
                html_content += f'<h3>Visualization:</h3><img src="annotated_images/{rel_path}" alt="Annotated {test_name}">'
            
            # Add Mermaid code if available
            if 'mermaid_content' in results:
                html_content += f'<h3>Generated Mermaid:</h3><div class="mermaid-code">{results["mermaid_content"]}</div>'
            
            # Add error if failed
            if not results.get('success', False) and 'error' in results:
                html_content += f'<h3>Error:</h3><p style="color: red;">{results["error"]}</p>'
            
            html_content += "</div>"
        
        html_content += """
</body>
</html>"""
        
        with open(summary_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Test summary created: {summary_path}")
        return summary_path


async def main():
    """Run comprehensive test suite"""
    test_folder = Path("tests")
    
    # Ensure test folder exists
    test_folder.mkdir(exist_ok=True)
    (test_folder / "images").mkdir(exist_ok=True)
    
    # Run comprehensive tests
    test_suite = ComprehensiveTestSuite(test_folder)
    results = await test_suite.run_comprehensive_tests()
    
    print(f"\n📁 All test outputs saved in: {test_folder / 'outputs'}")
    print(f"🎨 Check the following folders for results:")
    print(f"   • annotated_images/     - Visual analysis with bounding boxes")
    print(f"   • visual_comparisons/   - Side-by-side with statistics")
    print(f"   • mermaid_files/        - Generated Mermaid code")
    print(f"   • analysis_reports/     - Detailed JSON and text reports")
    print(f"   • test_summary_*.html   - Comprehensive HTML summary")


if __name__ == "__main__":
    asyncio.run(main())