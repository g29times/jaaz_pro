"""
Display Test Results - Show the visual outputs from the comprehensive test suite
"""

import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def display_test_results(test_folder: Path):
    """Display the test results with visualizations"""
    outputs_folder = test_folder / "outputs"
    
    if not outputs_folder.exists():
        print("❌ No test outputs found. Run comprehensive_test_suite.py first.")
        return
    
    print("🎨 Test Results Visualization")
    print("=" * 50)
    
    # Find annotated images
    annotated_folder = outputs_folder / "annotated_images"
    comparison_folder = outputs_folder / "visual_comparisons"
    
    annotated_images = list(annotated_folder.glob("*.png")) if annotated_folder.exists() else []
    comparison_images = list(comparison_folder.glob("*.png")) if comparison_folder.exists() else []
    
    print(f"📊 Found {len(annotated_images)} annotated images")
    print(f"📊 Found {len(comparison_images)} comparison images")
    
    # Display images
    if annotated_images or comparison_images:
        fig, axes = plt.subplots(2, max(len(annotated_images), 1), figsize=(15, 10))
        if len(annotated_images) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, img_path in enumerate(annotated_images):
            if i < len(axes[0]):
                img = Image.open(img_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Annotated: {img_path.stem}")
                axes[0, i].axis('off')
        
        for i, img_path in enumerate(comparison_images):
            if i < len(axes[1]):
                img = Image.open(img_path)
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"With Stats: {img_path.stem}")
                axes[1, i].axis('off')
        
        # Hide unused subplots
        for i in range(len(annotated_images), len(axes[0])):
            axes[0, i].axis('off')
        for i in range(len(comparison_images), len(axes[1])):
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(outputs_folder / "visualization_summary.png", dpi=150, bbox_inches='tight')
        print(f"✅ Visualization summary saved: {outputs_folder / 'visualization_summary.png'}")
        
        # Show the plot
        plt.show()
    
    # Show Mermaid files
    mermaid_folder = outputs_folder / "mermaid_files"
    if mermaid_folder.exists():
        mermaid_files = list(mermaid_folder.glob("*.mmd"))
        print(f"\n📄 Generated Mermaid Files ({len(mermaid_files)}):")
        
        for mmd_file in mermaid_files:
            print(f"\n🔍 {mmd_file.name}:")
            with open(mmd_file, 'r') as f:
                content = f.read()
                print("   " + content.replace('\n', '\n   '))
    
    # Show analysis reports
    reports_folder = outputs_folder / "analysis_reports"
    if reports_folder.exists():
        txt_reports = list(reports_folder.glob("*.txt"))
        print(f"\n📊 Analysis Reports ({len(txt_reports)}):")
        
        for report_file in txt_reports:
            print(f"\n📋 {report_file.name}:")
            with open(report_file, 'r') as f:
                lines = f.readlines()
                for line in lines[:20]:  # Show first 20 lines
                    print("   " + line.rstrip())
                if len(lines) > 20:
                    print(f"   ... ({len(lines) - 20} more lines)")
    
    # Show HTML summary info
    html_files = list(outputs_folder.glob("test_summary_*.html"))
    if html_files:
        print(f"\n🌐 HTML Summary Report:")
        print(f"   📄 {html_files[0].name}")
        print(f"   📁 Open this file in a web browser to see the complete visual report")
        print(f"   🔗 Full path: {html_files[0].absolute()}")


def show_file_structure(test_folder: Path):
    """Show the complete test folder structure"""
    print(f"\n📁 Test Folder Structure:")
    print(f"tests/")
    
    def print_tree(path, prefix=""):
        if path.is_dir():
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and item.name in ['outputs', 'images']:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(item, next_prefix)
    
    print_tree(test_folder)


if __name__ == "__main__":
    test_folder = Path("tests")
    
    if not test_folder.exists():
        print("❌ Test folder not found. Run the comprehensive test suite first.")
        sys.exit(1)
    
    show_file_structure(test_folder)
    display_test_results(test_folder)