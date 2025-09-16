"""
Enhanced Mermaid Generator - Phase 2 Intelligent Reconstruction
Uses computer vision analysis to recreate accurate flowcharts
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import math
import tempfile
from pathlib import Path

from ..models import ParsedElement, GeneratorOutput


@dataclass 
class FlowchartNode:
    """Represents a detected flowchart node"""
    id: str
    node_type: str  # start, end, process, decision
    text: str
    position: Tuple[int, int]  # (x, y)
    size: Tuple[int, int]      # (width, height)
    confidence: float


@dataclass
class FlowchartConnection:
    """Represents a connection between nodes"""
    from_node: str
    to_node: str
    label: Optional[str] = None
    confidence: float = 0.0


class IntelligentMermaidGenerator:
    """Generate accurate Mermaid diagrams from computer vision analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Node type classification thresholds
        self.oval_aspect_ratio_range = (0.8, 1.2)  # For start/end nodes
        self.diamond_size_threshold = 0.7  # For decision nodes
        self.rectangle_default = True  # Default to process nodes
        
        # Spatial analysis parameters
        self.connection_distance_threshold = 150  # Max distance for connections
        self.vertical_flow_threshold = 0.3  # Threshold for determining flow direction
        
    async def generate_from_visual_analysis(self, elements: List[ParsedElement], session_id: str) -> GeneratorOutput:
        """Generate Mermaid diagram from computer vision analysis"""
        self.logger.info(f"[{session_id}] Generating Mermaid from {len(elements)} visual elements")
        
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
                    "generator": "intelligent_mermaid",
                    "session_id": session_id,
                    "nodes_detected": len(nodes),
                    "connections_detected": len(connections),
                    "flow_direction": flow_direction,
                    "generation_method": "computer_vision_analysis",
                    "visual_elements_processed": len(elements)
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
        
        # Process shape elements
        shape_elements = [e for e in elements if e.element_type == "shape"]
        text_elements = [e for e in elements if e.element_type == "text"]
        
        for element in shape_elements:
            if not element.metadata or 'bounding_box' not in element.metadata:
                continue
                
            bbox = element.metadata['bounding_box']
            if len(bbox) < 4:
                continue
                
            x, y, w, h = bbox[:4]
            
            # Skip very small shapes
            if w < 30 or h < 20:
                continue
            
            # Classify node type based on shape and size
            shape_type = element.metadata.get('shape_type', 'rectangle')
            node_type = self._classify_node_type(shape_type, w, h, element.content)
            
            # Find associated text
            node_text = self._find_associated_text(x, y, w, h, text_elements)
            if not node_text:
                node_text = self._extract_text_from_content(element.content)
            
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
        
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Check content for keywords
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['start', 'begin']):
            return 'start'
        elif any(word in content_lower for word in ['end', 'stop', 'finish']):
            return 'end'
        elif any(word in content_lower for word in ['if', '?', 'decision', 'choice']):
            return 'decision'
        
        # Classify by shape
        if shape_type == 'circle' or (shape_type in ['polygon', 'rectangle'] and 0.8 <= aspect_ratio <= 1.2):
            # Round shapes are typically start/end
            if 'start' in content_lower or 'begin' in content_lower:
                return 'start'
            elif 'end' in content_lower or 'stop' in content_lower:
                return 'end'
            else:
                return 'start'  # Default for round shapes
        elif shape_type == 'polygon' and aspect_ratio < 0.8:
            # Diamond-like shapes are decisions
            return 'decision'
        else:
            # Rectangular shapes are processes
            return 'process'
    
    def _find_associated_text(self, x: int, y: int, w: int, h: int, text_elements: List[ParsedElement]) -> str:
        """Find text elements that are spatially associated with a shape"""
        associated_texts = []
        
        # Expand search area slightly to catch nearby text
        search_margin = 20
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
                
            tx, ty, tw, th = text_bbox[:4]
            text_center_x = tx + tw // 2
            text_center_y = ty + th // 2
            
            # Check if text center is within or near the shape
            if (expanded_x <= text_center_x <= expanded_x + expanded_w and 
                expanded_y <= text_center_y <= expanded_y + expanded_h):
                
                # Clean up the text content
                clean_text = text_elem.content.strip()
                if clean_text and len(clean_text) > 1:
                    associated_texts.append(clean_text)
        
        # Return the best text match
        if associated_texts:
            # Prefer shorter, more meaningful text
            associated_texts.sort(key=len)
            best_text = associated_texts[0]
            
            # Clean up common OCR artifacts
            best_text = best_text.replace('\n', ' ').replace('\t', ' ')
            best_text = ' '.join(best_text.split())  # Normalize whitespace
            
            # If text is too long, truncate
            if len(best_text) > 30:
                best_text = best_text[:27] + "..."
                
            return best_text
        
        return ""
    
    def _extract_text_from_content(self, content: str) -> str:
        """Extract meaningful text from element content description"""
        # Try to extract meaningful text from shape content
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
        text = re.sub(r'\b(rectangle|polygon|circle|shape)\b', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # If we have meaningful text, return it
        if len(text) > 2 and not text.isdigit():
            # Capitalize first letter
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
            return text[:20]  # Limit length
        
        # Default fallback based on position/context
        return "Process"
    
    def _analyze_connections(self, elements: List[ParsedElement], nodes: List[FlowchartNode], session_id: str) -> List[FlowchartConnection]:
        """Analyze spatial relationships to determine connections between nodes"""
        connections = []
        
        # Get arrow elements
        arrow_elements = [e for e in elements if e.element_type == "arrow"]
        
        self.logger.debug(f"[{session_id}] Analyzing {len(arrow_elements)} arrows for connections")
        
        for arrow in arrow_elements:
            if not arrow.metadata:
                continue
                
            # Get arrow endpoints
            start_point = arrow.metadata.get('start_point')
            end_point = arrow.metadata.get('end_point')
            
            if not start_point or not end_point:
                continue
            
            # Find closest nodes to arrow endpoints
            from_node = self._find_closest_node(start_point, nodes)
            to_node = self._find_closest_node(end_point, nodes)
            
            if from_node and to_node and from_node != to_node:
                connection = FlowchartConnection(
                    from_node=from_node.id,
                    to_node=to_node.id,
                    confidence=arrow.confidence
                )
                connections.append(connection)
                
                self.logger.debug(f"[{session_id}] Connected {from_node.id} -> {to_node.id}")
        
        # If no arrows detected, create logical connections based on spatial layout
        if not connections:
            connections = self._create_spatial_connections(nodes, session_id)
        
        return connections
    
    def _find_closest_node(self, point: Tuple[int, int], nodes: List[FlowchartNode]) -> Optional[FlowchartNode]:
        """Find the node closest to a given point"""
        if not nodes:
            return None
            
        min_distance = float('inf')
        closest_node = None
        
        px, py = point
        
        for node in nodes:
            nx, ny = node.position
            distance = math.sqrt((px - nx)**2 + (py - ny)**2)
            
            if distance < min_distance and distance < self.connection_distance_threshold:
                min_distance = distance
                closest_node = node
        
        return closest_node
    
    def _create_spatial_connections(self, nodes: List[FlowchartNode], session_id: str) -> List[FlowchartConnection]:
        """Create logical connections based on spatial layout when arrows aren't detected"""
        connections = []
        
        if len(nodes) < 2:
            return connections
        
        self.logger.info(f"[{session_id}] Creating spatial connections for {len(nodes)} nodes")
        
        # Sort nodes by vertical position (top to bottom)
        sorted_nodes = sorted(nodes, key=lambda n: n.position[1])
        
        # Create sequential connections
        for i in range(len(sorted_nodes) - 1):
            current = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]
            
            connection = FlowchartConnection(
                from_node=current.id,
                to_node=next_node.id,
                confidence=0.8
            )
            connections.append(connection)
        
        # Handle decision nodes - they might have multiple outputs
        decision_nodes = [n for n in nodes if n.node_type == 'decision']
        for decision in decision_nodes:
            # Find nodes at similar vertical level (parallel branches)
            decision_y = decision.position[1]
            parallel_nodes = [n for n in nodes if abs(n.position[1] - decision_y) < 100 and n != decision]
            
            # Connect decision to parallel nodes
            for target in parallel_nodes:
                if not any(c.from_node == decision.id and c.to_node == target.id for c in connections):
                    connection = FlowchartConnection(
                        from_node=decision.id,
                        to_node=target.id,
                        label="Yes" if target.position[0] > decision.position[0] else "No",
                        confidence=0.7
                    )
                    connections.append(connection)
        
        return connections
    
    def _determine_flow_direction(self, nodes: List[FlowchartNode], connections: List[FlowchartConnection]) -> str:
        """Determine the primary flow direction of the flowchart"""
        if not nodes:
            return "TD"  # Top-down default
        
        # Calculate vertical vs horizontal spread
        positions = [n.position for n in nodes]
        
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        
        horizontal_spread = max_x - min_x
        vertical_spread = max_y - min_y
        
        # Determine primary direction
        if vertical_spread > horizontal_spread * 1.5:
            return "TD"  # Top-down
        elif horizontal_spread > vertical_spread * 1.5:
            return "LR"  # Left-right
        else:
            return "TD"  # Default to top-down
    
    def _generate_mermaid_code(self, nodes: List[FlowchartNode], connections: List[FlowchartConnection], 
                             direction: str, session_id: str) -> str:
        """Generate Mermaid code from nodes and connections"""
        
        lines = [f"flowchart {direction}"]
        
        # Generate node definitions
        node_map = {}  # Map node IDs to Mermaid node IDs
        mermaid_id = ord('A')
        
        for node in nodes:
            mermaid_node_id = chr(mermaid_id)
            node_map[node.id] = mermaid_node_id
            
            # Format node text
            text = node.text if node.text else "Step"
            text = text.replace('"', "'")  # Escape quotes
            
            # Choose node shape based on type
            if node.node_type == 'start':
                lines.append(f"    {mermaid_node_id}([{text}])")
            elif node.node_type == 'end':
                lines.append(f"    {mermaid_node_id}([{text}])")
            elif node.node_type == 'decision':
                lines.append(f"    {mermaid_node_id}{{{text}}}")
            else:  # process
                lines.append(f"    {mermaid_node_id}[{text}]")
            
            mermaid_id += 1
        
        # Generate connections with deduplication
        connection_set = set()  # Use set to avoid duplicates
        
        for connection in connections:
            from_id = node_map.get(connection.from_node)
            to_id = node_map.get(connection.to_node)
            
            if from_id and to_id and from_id != to_id:
                # Create connection string for deduplication
                if connection.label:
                    conn_str = f"    {from_id} -->|{connection.label}| {to_id}"
                else:
                    conn_str = f"    {from_id} --> {to_id}"
                
                connection_set.add(conn_str)
        
        # Add unique connections to lines
        lines.extend(sorted(connection_set))  # Sort for consistent output
        
        mermaid_code = "\n".join(lines)
        
        self.logger.info(f"[{session_id}] Generated Mermaid with {len(nodes)} nodes and {len(connections)} connections")
        self.logger.debug(f"[{session_id}] Mermaid code:\n{mermaid_code}")
        
        return mermaid_code
    
    def _create_output_file(self, mermaid_code: str, session_id: str) -> Path:
        """Create output file for the generated Mermaid code"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intelligent_flowchart_{session_id}_{timestamp}.mmd"
        
        output_path = Path(tempfile.gettempdir()) / filename
        
        with open(output_path, 'w') as f:
            f.write(mermaid_code)
        
        return output_path
    
    async def _generate_fallback(self, elements: List[ParsedElement], session_id: str) -> GeneratorOutput:
        """Fallback generation when intelligent analysis fails"""
        self.logger.warning(f"[{session_id}] Using fallback generation")
        
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
                "generator": "intelligent_mermaid_fallback",
                "session_id": session_id,
                "generation_method": "fallback"
            }
        )