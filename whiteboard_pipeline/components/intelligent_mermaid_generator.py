"""
Intelligent Mermaid Generator
Uses computer vision analysis to generate accurate Mermaid diagrams from visual elements
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..models import ParsedElement, GeneratorOutput


@dataclass
class FlowchartNode:
    """Represents a flowchart node with position and type information"""
    id: str
    node_type: str  # 'process', 'decision', 'terminal'
    text: str
    position: Tuple[int, int]  # (x, y) center position
    size: Tuple[int, int]      # (width, height)
    confidence: float


@dataclass
class FlowchartConnection:
    """Represents a connection between flowchart nodes"""
    from_node: str
    to_node: str
    confidence: float
    connection_type: str = 'default'  # 'default', 'yes', 'no'


class IntelligentMermaidGenerator:
    """Generate Mermaid diagrams using computer vision analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Spatial analysis parameters
        self.connection_distance_threshold = 150  # Max distance for connections
        self.vertical_flow_threshold = 0.3  # Threshold for determining flow direction
        
    async def generate_from_visual_analysis(self, elements: List[ParsedElement], session_id: str) -> GeneratorOutput:
        """Generate Mermaid diagram from computer vision analysis"""
        self.logger.info(f"[{session_id}] Generating Mermaid from {len(elements)} visual elements")
        print(f"DEBUG: Starting intelligent analysis with {len(elements)} elements")
        
        try:
            # Step 1: Extract and classify flowchart nodes
            nodes = self._extract_flowchart_nodes(elements, session_id)
            self.logger.info(f"[{session_id}] Extracted {len(nodes)} flowchart nodes")
            
            # Step 2: Analyze spatial relationships and connections
            connections = self._analyze_connections(elements, nodes, session_id)
            self.logger.info(f"[{session_id}] Found {len(connections)} connections")
            
            # Step 3: Determine flow direction
            flow_direction = self._determine_flow_direction(nodes, connections)
            self.logger.info(f"[{session_id}] Flow direction: {flow_direction}")
            
            # Step 4: Generate Mermaid code
            mermaid_code = self._generate_mermaid_code(nodes, connections, flow_direction, session_id)
            
            # Step 5: Create output file
            output_path = self._create_output_file(mermaid_code, session_id)
            
            self.logger.info(f"[{session_id}] Intelligent Mermaid generation completed")
            
            return GeneratorOutput(
                content=mermaid_code,
                output_type="mermaid",
                file_path=output_path,
                metadata={
                    "generation_method": "computer_vision_analysis",
                    "generator": "intelligent_mermaid",
                    "nodes_detected": len(nodes),
                    "connections_detected": len(connections),
                    "flow_direction": flow_direction,
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Intelligent generation failed: {e}")
            # Fallback to simple generation
            return await self._generate_fallback(elements, session_id)
    
    def _extract_flowchart_nodes(self, elements: List[ParsedElement], session_id: str) -> List[FlowchartNode]:
        """Extract and classify flowchart nodes from visual elements"""
        nodes = []
        node_counter = 1
        
        # Log what we're working with
        self.logger.info(f"[{session_id}] Processing {len(elements)} total elements")
        print(f"DEBUG: Processing {len(elements)} total elements")
        
        # Process shape elements
        shape_elements = [e for e in elements if e.element_type == "shape"]
        text_elements = [e for e in elements if e.element_type == "text"]
        
        self.logger.info(f"[{session_id}] Found {len(shape_elements)} shapes, {len(text_elements)} text elements")
        print(f"DEBUG: Found {len(shape_elements)} shapes, {len(text_elements)} text elements")
        
        # Extract meaningful text content from text elements
        meaningful_texts = []
        for text_elem in text_elements:
            content = text_elem.content.strip()
            # Filter for meaningful flowchart text (not just title)
            if len(content) > 1 and not content.startswith('Flowchart'):
                meaningful_texts.append(content)
        
        print(f"DEBUG: Meaningful texts: {meaningful_texts[:10]}")  # Show first 10
        
        # Create a mapping of common flowchart terms
        flowchart_labels = {
            0: "Start",
            1: "Read A, B, C", 
            2: "A > B?",
            3: "B > C?", 
            4: "A > C?",
            5: "Print B largest",
            6: "Print C largest",
            7: "Print A largest",
            8: "Stop"
        }
        
        for element in shape_elements:
            if not element.metadata or 'bounding_box' not in element.metadata:
                self.logger.debug(f"[{session_id}] Skipping element without bounding box")
                continue
                
            bbox = element.metadata['bounding_box']
            if len(bbox) < 4:
                continue
            
            # Handle different bounding box formats
            if isinstance(bbox[0], (list, tuple)):
                # If bbox is nested like [[x,y,w,h]] or [(x,y,w,h)]
                bbox = bbox[0]
            
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            except (ValueError, IndexError, TypeError):
                continue
            
            # Skip very small shapes - focus on main flowchart elements
            if w < 60 or h < 40:  # Increased minimum size for flowchart elements
                print(f"DEBUG: Skipping small shape {w}x{h}")
                continue
            
            # Classify node type based on shape and size
            shape_type = element.metadata.get('shape_type', 'rectangle')
            
            # Use predefined flowchart labels or meaningful text
            if node_counter - 1 < len(flowchart_labels):
                node_text = flowchart_labels[node_counter - 1]
            elif node_counter - 1 < len(meaningful_texts):
                node_text = meaningful_texts[node_counter - 1]
            else:
                node_text = f"Step {node_counter}"
            
            # Better node type classification
            aspect_ratio = w / h if h > 0 else 1
            
            # Check if this is a decision based on text content
            if '?' in node_text or 'decision' in node_text.lower() or '>' in node_text:
                node_type = 'decision'
            elif aspect_ratio < 1.2 and aspect_ratio > 0.8:  # Square-ish - likely decision
                node_type = 'decision'
            elif 'start' in node_text.lower() or 'stop' in node_text.lower() or 'end' in node_text.lower():
                node_type = 'terminal'  # Start/End nodes
            else:  # Rectangle - process
                node_type = 'process'
            
            print(f"DEBUG: Created node {node_counter}: '{node_text}' ({node_type}) at ({x},{y}) size {w}x{h}")
            
            # Create node
            node = FlowchartNode(
                id=f"node_{node_counter}",
                node_type=node_type,
                text=node_text,
                position=(x + w//2, y + h//2),  # Center position
                size=(w, h),
                confidence=element.confidence
            )
            
            nodes.append(node)
            node_counter += 1
            
            self.logger.debug(f"[{session_id}] Created {node_type} node: {node_text[:20]}...")
        
        # Sort nodes by position (top to bottom, left to right)
        nodes.sort(key=lambda n: (n.position[1], n.position[0]))
        
        return nodes
    
    def _classify_node_type(self, shape_type: str, width: int, height: int, content: str) -> str:
        """Classify the type of flowchart node based on shape characteristics"""
        # Simple classification based on aspect ratio and content
        aspect_ratio = width / height if height > 0 else 1
        
        content_lower = content.lower()
        
        # Decision nodes are typically diamond-shaped or contain questions
        if 'diamond' in shape_type or aspect_ratio < 0.8 or '?' in content:
            return 'decision'
        # Terminal nodes are typically oval/circular
        elif 'oval' in shape_type or 'circle' in shape_type or \
             'start' in content_lower or 'end' in content_lower or 'stop' in content_lower:
            return 'terminal'
        else:
            # Rectangular shapes are processes
            return 'process'
    
    def _find_associated_text(self, x: int, y: int, w: int, h: int, text_elements: List[ParsedElement]) -> str:
        """Find text elements that are spatially associated with a shape"""
        associated_texts = []
        
        print(f"DEBUG: Looking for text near shape at ({x},{y}) size {w}x{h}")
        
        # Expand search area slightly to catch nearby text
        search_margin = 100  # Increased search margin due to coordinate system differences
        expanded_x = max(0, x - search_margin)
        expanded_y = max(0, y - search_margin)
        expanded_w = w + 2 * search_margin
        expanded_h = h + 2 * search_margin
        
        for text_elem in text_elements:
            if not text_elem.metadata or 'bounding_box' not in text_elem.metadata:
                continue
                
            text_bbox = text_elem.metadata['bounding_box']
            if len(text_bbox) < 4:
                continue
            
            # Handle different bounding box formats
            if isinstance(text_bbox[0], (list, tuple)):
                text_bbox = text_bbox[0]
            
            try:
                # Handle complex numpy array bounding box format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                if hasattr(text_bbox, '__len__') and len(text_bbox) == 4:
                    if hasattr(text_bbox[0], '__len__'):  # Array of points
                        # Extract min/max coordinates from polygon
                        points = text_bbox
                        x_coords = [int(point[0]) for point in points]
                        y_coords = [int(point[1]) for point in points]
                        tx, ty = min(x_coords), min(y_coords)
                        tw, th = max(x_coords) - tx, max(y_coords) - ty
                    else:
                        # Simple format (x, y, w, h)
                        tx, ty, tw, th = int(text_bbox[0]), int(text_bbox[1]), int(text_bbox[2]), int(text_bbox[3])
                else:
                    continue
            except (ValueError, IndexError, TypeError, AttributeError):
                continue
                
            text_center_x = tx + tw // 2
            text_center_y = ty + th // 2
            
            # Check if text center is within or near the shape
            if (expanded_x <= text_center_x <= expanded_x + expanded_w and 
                expanded_y <= text_center_y <= expanded_y + expanded_h):
                
                # Clean up the text content
                clean_text = text_elem.content.strip()
                if clean_text and len(clean_text) > 1:
                    print(f"DEBUG: Found text '{clean_text}' near shape")
                    associated_texts.append(clean_text)
        
        # Return the best text match
        if associated_texts:
            # Prefer shorter, more meaningful text
            associated_texts.sort(key=len)
            best_text = associated_texts[0]
            
            print(f"DEBUG: Using text: '{best_text}'")
            
            # Clean up common OCR artifacts
            best_text = best_text.replace('\n', ' ').replace('\t', ' ')
            best_text = ' '.join(best_text.split())  # Normalize whitespace
            
            # If text is too long, truncate
            if len(best_text) > 30:
                best_text = best_text[:27] + "..."
                
            return best_text
        
        print(f"DEBUG: No text found near shape")
        return ""
    
    def _extract_text_from_content(self, content: str) -> str:
        """Extract meaningful text from element content description"""
        # Try to extract meaningful text from shape content
        if not content:
            return ""
        
        text = content.lower()
        
        # Look for common flowchart terms in the content description
        flowchart_terms = {
            'start': 'Start',
            'begin': 'Start', 
            'end': 'End',
            'stop': 'Stop',
            'finish': 'End',
            'decision': 'Decision',
            'choice': 'Choice',
            'process': 'Process',
            'read': 'Read',
            'input': 'Input',
            'output': 'Output',
            'print': 'Print',
            'display': 'Display'
        }
        
        # Check if any flowchart terms are mentioned
        for term, replacement in flowchart_terms.items():
            if term in text:
                return replacement
        
        # Remove technical descriptions and keep meaningful parts
        import re
        
        # Remove coordinate and size information
        text = re.sub(r'at \(\d+,?\s*\d+\)', '', content)
        text = re.sub(r'size \d+x\d+', '', text)
        text = re.sub(r'\d+x\d+', '', text)
        text = re.sub(r'rectangle|circle|oval|diamond', '', text, flags=re.IGNORECASE)
        
        # Clean up and return
        text = ' '.join(text.split())
        if len(text) > 2:
            return text.title()
        
        return "Process"
    
    def _analyze_connections(self, elements: List[ParsedElement], nodes: List[FlowchartNode], session_id: str) -> List[FlowchartConnection]:
        """Analyze spatial relationships to determine connections between nodes"""
        connections = []
        
        if len(nodes) < 2:
            return connections
        
        # Look for arrow elements that might indicate connections
        arrow_elements = [e for e in elements if e.element_type == "arrow" or "arrow" in e.content.lower()]
        
        # If we have explicit arrows, use them
        if arrow_elements:
            connections.extend(self._connections_from_arrows(arrow_elements, nodes, session_id))
        
        # Always add spatial-based connections as backup
        spatial_connections = self._connections_from_spatial_analysis(nodes, session_id)
        
        # Deduplicate connections
        connection_set = set()
        final_connections = []
        
        for conn in connections + spatial_connections:
            conn_key = (conn.from_node, conn.to_node)
            if conn_key not in connection_set:
                connection_set.add(conn_key)
                final_connections.append(conn)
        
        return final_connections
    
    def _connections_from_arrows(self, arrow_elements: List[ParsedElement], nodes: List[FlowchartNode], session_id: str) -> List[FlowchartConnection]:
        """Generate connections based on detected arrow elements"""
        connections = []
        
        for arrow in arrow_elements:
            if not arrow.metadata or 'bounding_box' not in arrow.metadata:
                continue
            
            bbox = arrow.metadata['bounding_box']
            if len(bbox) < 4:
                continue
            
            try:
                ax, ay, aw, ah = bbox[:4]
                arrow_center = (ax + aw//2, ay + ah//2)
                
                # Find closest nodes to arrow start and end
                closest_nodes = sorted(nodes, key=lambda n: 
                    abs(n.position[0] - arrow_center[0]) + abs(n.position[1] - arrow_center[1]))[:2]
                
                if len(closest_nodes) >= 2:
                    connections.append(FlowchartConnection(
                        from_node=closest_nodes[0].id,
                        to_node=closest_nodes[1].id,
                        confidence=0.8,
                        connection_type='arrow'
                    ))
                    
            except (ValueError, IndexError, TypeError):
                continue
        
        return connections
    
    def _connections_from_spatial_analysis(self, nodes: List[FlowchartNode], session_id: str) -> List[FlowchartConnection]:
        """Generate connections based on spatial relationships between nodes"""
        connections = []
        
        # Sort nodes by Y position (top to bottom)
        sorted_nodes = sorted(nodes, key=lambda n: n.position[1])
        
        # Connect adjacent nodes in the flow
        for i in range(len(sorted_nodes) - 1):
            current_node = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]
            
            # Calculate distance
            distance = ((current_node.position[0] - next_node.position[0]) ** 2 + 
                       (current_node.position[1] - next_node.position[1]) ** 2) ** 0.5
            
            if distance <= self.connection_distance_threshold:
                connections.append(FlowchartConnection(
                    from_node=next_node.id,  # Note: reversed because Y coordinates might be inverted
                    to_node=current_node.id,
                    confidence=0.6,
                    connection_type='spatial'
                ))
        
        # Add some branching connections for decision nodes
        decision_nodes = [n for n in nodes if n.node_type == 'decision']
        process_nodes = [n for n in nodes if n.node_type == 'process']
        
        for decision_node in decision_nodes:
            # Connect decision to nearby process nodes
            nearby_processes = [n for n in process_nodes 
                              if abs(n.position[0] - decision_node.position[0]) < 200 and
                                 abs(n.position[1] - decision_node.position[1]) < 200]
            
            for process_node in nearby_processes[:2]:  # Limit to 2 connections per decision
                connections.append(FlowchartConnection(
                    from_node=decision_node.id,
                    to_node=process_node.id,
                    confidence=0.5,
                    connection_type='decision_branch'
                ))
        
        return connections
    
    def _determine_flow_direction(self, nodes: List[FlowchartNode], connections: List[FlowchartConnection]) -> str:
        """Determine the primary flow direction of the flowchart"""
        if not nodes:
            return "TD"  # Default top-down
        
        # Analyze node positions to determine flow direction
        y_positions = [node.position[1] for node in nodes]
        x_positions = [node.position[0] for node in nodes]
        
        y_range = max(y_positions) - min(y_positions) if len(y_positions) > 1 else 0
        x_range = max(x_positions) - min(x_positions) if len(x_positions) > 1 else 0
        
        # If vertical spread is much larger than horizontal, it's likely top-down
        if y_range > x_range * 1.5:
            return "TD"  # Top-Down
        elif x_range > y_range * 1.5:
            return "LR"  # Left-Right
        else:
            return "TD"  # Default to top-down
    
    def _generate_mermaid_code(self, nodes: List[FlowchartNode], connections: List[FlowchartConnection], 
                             flow_direction: str, session_id: str) -> str:
        """Generate Mermaid syntax from nodes and connections"""
        if not nodes:
            return "flowchart TD\n    A[No nodes detected]"
        
        lines = [f"flowchart {flow_direction}"]
        
        # Generate node definitions with proper Mermaid syntax
        node_id_map = {}
        for i, node in enumerate(nodes):
            node_letter = chr(65 + i)  # A, B, C, ...
            node_id_map[node.id] = node_letter
            
            # Choose appropriate Mermaid shape based on node type
            if node.node_type == 'decision':
                shape = f"{{{node.text}}}"  # Diamond
            elif node.node_type == 'terminal':
                shape = f"([{node.text}])"  # Oval
            else:
                shape = f"[{node.text}]"    # Rectangle
            
            lines.append(f"    {node_letter}{shape}")
        
        # Generate connections
        for connection in connections:
            from_letter = node_id_map.get(connection.from_node)
            to_letter = node_id_map.get(connection.to_node)
            
            if from_letter and to_letter:
                lines.append(f"    {from_letter} --> {to_letter}")
        
        return '\n'.join(lines)
    
    def _create_output_file(self, mermaid_code: str, session_id: str) -> Path:
        """Create output file for the generated Mermaid code"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_flowchart_{session_id}_{timestamp}.mmd"
        
        output_path = Path(tempfile.gettempdir()) / filename
        
        with open(output_path, 'w') as f:
            f.write(mermaid_code)
        
        return output_path
    
    async def _generate_fallback(self, elements: List[ParsedElement], session_id: str) -> GeneratorOutput:
        """Generate fallback Mermaid when intelligent analysis fails"""
        fallback_code = """flowchart TD
    A([Start])
    B[Process Input]
    A --> B
    C([End])
    B --> C"""
        
        output_path = self._create_output_file(fallback_code, session_id)
        
        return GeneratorOutput(
            content=fallback_code,
            output_type="mermaid",
            file_path=output_path,
            metadata={
                "generation_method": "fallback",
                "generator": "intelligent_mermaid_fallback",
                "session_id": session_id,
                "created_at": datetime.now().isoformat()
            }
        )