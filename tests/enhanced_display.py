"""
Enhanced Test Results Display with Mermaid Visualization
Shows visual outputs and renders Mermaid flowcharts
"""

import sys
import json
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class MermaidRenderer:
    """Render Mermaid diagrams to images"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir())
        
    def render_mermaid_to_image(self, mermaid_code: str, output_path: Path) -> bool:
        """Render Mermaid diagram to PNG image"""
        try:
            # Try using mermaid-cli if available
            temp_mmd = self.temp_dir / f"temp_{output_path.stem}.mmd"
            
            # Write Mermaid code to temporary file
            with open(temp_mmd, 'w') as f:
                f.write(mermaid_code)
            
            # Try to render with mermaid-cli
            try:
                result = subprocess.run([
                    'mmdc', '-i', str(temp_mmd), '-o', str(output_path),
                    '--theme', 'default', '--backgroundColor', 'white'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and output_path.exists():
                    temp_mmd.unlink()  # Clean up
                    return True
                    
            except (subprocess.SubprocessError, FileNotFoundError):
                pass  # Fall back to manual rendering
            
            # Fallback: Create a simple text-based representation
            self._create_text_diagram(mermaid_code, output_path)
            temp_mmd.unlink()  # Clean up
            return True
            
        except Exception as e:
            print(f"Warning: Could not render Mermaid diagram: {e}")
            return False
    
    def _create_text_diagram(self, mermaid_code: str, output_path: Path):
        """Create a simple text-based diagram representation"""
        lines = mermaid_code.strip().split('\n')
        
        # Create image canvas
        img_width, img_height = 800, 600
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a better font
            font = ImageFont.truetype("Arial.ttf", 14)
            title_font = ImageFont.truetype("Arial.ttf", 18)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        draw.text((20, 20), "Generated Mermaid Flowchart", fill='black', font=title_font)
        
        # Parse and visualize Mermaid elements
        y_offset = 60
        nodes = {}
        connections = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('flowchart'):
                continue
                
            if '-->' in line:
                # Connection line
                parts = line.split('-->')
                if len(parts) == 2:
                    from_node = parts[0].strip()
                    to_node = parts[1].strip()
                    connections.append((from_node, to_node))
            else:
                # Node definition
                if any(bracket in line for bracket in ['([', '])', '[', ']', '{', '}']):
                    # Extract node info
                    node_id = line.split('[')[0].split('(')[0].split('{')[0].strip()
                    if node_id and node_id not in nodes:
                        nodes[node_id] = {
                            'text': line,
                            'x': 100 + len(nodes) * 150,
                            'y': y_offset + (len(nodes) % 3) * 80
                        }
        
        # Draw nodes
        for node_id, node_info in nodes.items():
            x, y = node_info['x'], node_info['y']
            
            # Determine shape based on brackets
            text = node_info['text']
            if '([' in text and '])' in text:
                # Oval (start/end)
                draw.ellipse([x-50, y-20, x+50, y+20], outline='black', fill='lightblue', width=2)
            elif '[' in text and ']' in text:
                # Rectangle (process)
                draw.rectangle([x-50, y-20, x+50, y+20], outline='black', fill='lightgreen', width=2)
            elif '{' in text and '}' in text:
                # Diamond (decision) - approximated as rectangle
                draw.rectangle([x-40, y-15, x+40, y+15], outline='black', fill='lightyellow', width=2)
            
            # Add node label
            label = node_id
            if '(' in text:
                content = text.split('(')[1].split(')')[0] if ')' in text else ''
                if content:
                    label = content
            elif '[' in text:
                content = text.split('[')[1].split(']')[0] if ']' in text else ''
                if content:
                    label = content
            
            # Draw text (truncate if too long)
            if len(label) > 10:
                label = label[:10] + '...'
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text((x - text_width//2, y - text_height//2), label, fill='black', font=font)
        
        # Draw connections
        for from_node, to_node in connections:
            if from_node in nodes and to_node in nodes:
                x1, y1 = nodes[from_node]['x'], nodes[from_node]['y']
                x2, y2 = nodes[to_node]['x'], nodes[to_node]['y']
                
                # Draw arrow
                draw.line([x1+50, y1, x2-50, y2], fill='black', width=2)
                
                # Draw arrowhead
                if x2 > x1:
                    draw.polygon([x2-50, y2, x2-40, y2-5, x2-40, y2+5], fill='black')
        
        # Add code representation at bottom
        code_y = img_height - 150
        draw.text((20, code_y), "Mermaid Code:", fill='black', font=title_font)
        code_y += 25
        
        for i, line in enumerate(lines[:8]):  # Show first 8 lines
            if code_y < img_height - 20:
                draw.text((20, code_y), line, fill='gray', font=font)
                code_y += 15
        
        if len(lines) > 8:
            draw.text((20, code_y), f"... ({len(lines) - 8} more lines)", fill='gray', font=font)
        
        # Save image
        img.save(output_path)


class EnhancedTestDisplay:
    """Enhanced test results display with Mermaid visualization"""
    
    def __init__(self, test_folder: Path):
        self.test_folder = test_folder
        self.outputs_folder = test_folder / "outputs"
        self.renderer = MermaidRenderer()
        
        # Create mermaid renders folder
        self.mermaid_renders_folder = self.outputs_folder / "mermaid_renders"
        self.mermaid_renders_folder.mkdir(exist_ok=True)
    
    def display_comprehensive_results(self):
        """Display all test results with Mermaid visualizations"""
        print("🎨 Enhanced Test Results with Mermaid Visualization")
        print("=" * 60)
        
        if not self.outputs_folder.exists():
            print("❌ No test outputs found. Run comprehensive_test_suite.py first.")
            return
        
        # Show folder structure
        self._show_test_structure()
        
        # Process and display Mermaid files
        self._process_mermaid_files()
        
        # Display analysis results
        self._display_analysis_results()
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization()
        
        # Show HTML report info
        self._show_html_reports()
    
    def _show_test_structure(self):
        """Show the organized test folder structure"""
        print("\n📁 Organized Test Folder Structure:")
        print("tests/")
        
        def print_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return
                
            if path.is_dir():
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    print(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and current_depth < max_depth - 1:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        print_tree(item, next_prefix, max_depth, current_depth + 1)
        
        print_tree(self.test_folder)
    
    def _process_mermaid_files(self):
        """Process and visualize Mermaid files"""
        mermaid_folder = self.outputs_folder / "mermaid_files"
        
        if not mermaid_folder.exists():
            print("\n⚠️  No Mermaid files found")
            return
        
        mermaid_files = list(mermaid_folder.glob("*.mmd"))
        print(f"\n🎯 Processing {len(mermaid_files)} Mermaid Files:")
        
        rendered_images = []
        
        for mmd_file in mermaid_files:
            print(f"\n📄 Processing: {mmd_file.name}")
            
            # Read Mermaid content
            with open(mmd_file, 'r') as f:
                mermaid_content = f.read()
            
            print("   Mermaid Code:")
            for i, line in enumerate(mermaid_content.split('\n'), 1):
                print(f"   {i:2d}: {line}")
            
            # Render to image
            render_path = self.mermaid_renders_folder / f"{mmd_file.stem}_rendered.png"
            success = self.renderer.render_mermaid_to_image(mermaid_content, render_path)
            
            if success:
                print(f"   ✅ Rendered to: {render_path.name}")
                rendered_images.append(render_path)
            else:
                print(f"   ⚠️  Could not render diagram")
        
        return rendered_images
    
    def _display_analysis_results(self):
        """Display detailed analysis results"""
        reports_folder = self.outputs_folder / "analysis_reports"
        
        if not reports_folder.exists():
            return
        
        json_reports = list(reports_folder.glob("*.json"))
        print(f"\n📊 Analysis Results ({len(json_reports)} reports):")
        
        for report_file in json_reports:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            print(f"\n🔍 {report_data.get('test_name', 'Unknown Test')}:")
            
            summary = report_data.get('summary', {})
            print(f"   Success: {'✅' if summary.get('success') else '❌'}")
            print(f"   Processing time: {summary.get('processing_time', 0):.3f}s")
            print(f"   Elements detected: {summary.get('elements_detected', 0)}")
            
            # Show element breakdown
            results = report_data.get('results', {})
            element_stats = results.get('element_stats', {})
            if element_stats:
                print(f"   Element breakdown:")
                for elem_type, count in element_stats.items():
                    print(f"     • {elem_type}: {count}")
    
    def _create_comprehensive_visualization(self):
        """Create a comprehensive visualization combining all results"""
        print(f"\n🎨 Creating Comprehensive Visualization...")
        
        # Collect all images
        annotated_images = list((self.outputs_folder / "annotated_images").glob("*.png")) if (self.outputs_folder / "annotated_images").exists() else []
        comparison_images = list((self.outputs_folder / "visual_comparisons").glob("*.png")) if (self.outputs_folder / "visual_comparisons").exists() else []
        mermaid_renders = list(self.mermaid_renders_folder.glob("*.png"))
        
        total_images = len(annotated_images) + len(comparison_images) + len(mermaid_renders)
        
        if total_images == 0:
            print("   ⚠️  No images to display")
            return
        
        # Create subplot layout
        cols = min(3, total_images)
        rows = (total_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if total_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Add annotated images
        for img_path in annotated_images:
            if plot_idx < len(axes):
                img = Image.open(img_path)
                axes[plot_idx].imshow(img)
                axes[plot_idx].set_title(f"Computer Vision Analysis\n{img_path.stem}")
                axes[plot_idx].axis('off')
                plot_idx += 1
        
        # Add comparison images
        for img_path in comparison_images:
            if plot_idx < len(axes):
                img = Image.open(img_path)
                axes[plot_idx].imshow(img)
                axes[plot_idx].set_title(f"Analysis with Statistics\n{img_path.stem}")
                axes[plot_idx].axis('off')
                plot_idx += 1
        
        # Add Mermaid renders
        for img_path in mermaid_renders:
            if plot_idx < len(axes):
                img = Image.open(img_path)
                axes[plot_idx].imshow(img)
                axes[plot_idx].set_title(f"Generated Mermaid\n{img_path.stem}")
                axes[plot_idx].axis('off')
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        output_path = self.outputs_folder / "comprehensive_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f"   ✅ Comprehensive visualization saved: {output_path.name}")
        
        # Show the plot
        plt.show()
        
        return output_path
    
    def _show_html_reports(self):
        """Show information about HTML reports"""
        html_files = list(self.outputs_folder.glob("test_summary_*.html"))
        
        print(f"\n🌐 HTML Dashboard:")
        if html_files:
            for html_file in html_files:
                print(f"   📄 {html_file.name}")
                print(f"   🔗 file://{html_file.absolute()}")
                print(f"   💡 Open in web browser for interactive dashboard")
        else:
            print("   ⚠️  No HTML reports found")
    
    def show_mermaid_renders(self):
        """Display just the Mermaid renders"""
        mermaid_renders = list(self.mermaid_renders_folder.glob("*.png"))
        
        print(f"\n🎯 Mermaid Flowchart Visualizations ({len(mermaid_renders)}):")
        
        for render_path in mermaid_renders:
            print(f"\n📊 {render_path.name}:")
            
            # Display the image
            try:
                img = Image.open(render_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.title(f"Mermaid Visualization: {render_path.stem}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
                print(f"   ✅ Displayed: {render_path.name}")
                
            except Exception as e:
                print(f"   ❌ Could not display: {e}")


def main():
    """Main function to display enhanced test results"""
    test_folder = Path(__file__).parent
    
    if not test_folder.exists():
        print("❌ Test folder not found.")
        return
    
    display = EnhancedTestDisplay(test_folder)
    
    # Show comprehensive results
    display.display_comprehensive_results()
    
    # Show Mermaid renders specifically
    print("\n" + "="*60)
    display.show_mermaid_renders()
    
    print(f"\n{'='*60}")
    print(f"✅ Enhanced test visualization complete!")
    print(f"📁 All outputs saved in: {test_folder / 'outputs'}")
    print(f"🎨 Visual results available in multiple formats")
    print(f"🎯 Mermaid flowcharts rendered in: mermaid_renders/")


if __name__ == "__main__":
    main()