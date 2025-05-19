import re
import math
import logging
import traceback
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, Arc
import numpy as np
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_coordinates_from_path(path_data):
    # Split the path data into commands and their parameters
    commands = path_data.strip().split()
    if not commands:
        return []
        
    # Convert to absolute coordinates
    current_x, current_y = 0, 0
    absolute_coords = []
    i = 0
    
    while i < len(commands):
        cmd = commands[i].upper()
        try:
            if cmd == 'M':
                if i + 2 >= len(commands):
                    break
                # Handle coordinates that might be comma-separated
                coords = commands[i+1].split(',')
                if len(coords) == 2:
                    current_x, current_y = float(coords[0]), float(coords[1])
                else:
                    current_x, current_y = float(commands[i+1]), float(commands[i+2])
                    i += 1
                absolute_coords.append(('M', current_x, current_y))
                i += 2
                
            elif cmd == 'L':
                if i + 2 >= len(commands):
                    break
                # Handle coordinates that might be comma-separated
                coords = commands[i+1].split(',')
                if len(coords) == 2:
                    current_x, current_y = float(coords[0]), float(coords[1])
                else:
                    current_x, current_y = float(commands[i+1]), float(commands[i+2])
                    i += 1
                absolute_coords.append(('L', current_x, current_y))
                i += 2
                
            elif cmd == 'A':
                if i + 7 >= len(commands):
                    break
                    
                # Extract arc parameters
                rx = float(commands[i+1])
                ry = float(commands[i+2])
                x_axis_rotation = float(commands[i+3])
                large_arc_flag = int(commands[i+4])
                sweep_flag = int(commands[i+5])
                end_x = float(commands[i+6])
                end_y = float(commands[i+7])
                
                # Add the arc command with all its parameters
                absolute_coords.append(('A', rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y))
                
                current_x, current_y = end_x, end_y
                i += 8
                
            else:
                i += 1
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing command {cmd}: {e}")
            i += 1
            continue
    
    return absolute_coords

def calculate_bounding_box(elem):
    """
    Calculate the bounding box of an SVG element using svgpathtools.
    """
    try:
        if elem.tag.endswith('path'):
            # Convert element to string and parse with svgpathtools
            path_str = ET.tostring(elem, encoding='unicode')
            paths, attributes = svg2paths(path_str)
            
            if not paths:
                logger.warning(f"No paths found in element")
                return 0, 0, 0, 0
                
            # Get bounding box of all paths
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            for path in paths:
                bbox = path.bbox()
                if bbox:
                    min_x = min(min_x, bbox[0].real)
                    min_y = min(min_y, bbox[0].imag)
                    max_x = max(max_x, bbox[1].real)
                    max_y = max(max_y, bbox[1].imag)
            
            if min_x == float('inf'):
                logger.warning(f"Invalid bounding box calculated")
                return 0, 0, 0, 0
                
            # Add padding
            padding = 2.0
            bbox = min_x - padding, min_y - padding, max_x + padding, max_y + padding
            logger.debug(f"Path bbox: {bbox}")
            return bbox
            
        elif elem.tag.endswith('circle'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            padding = 2.0
            bbox = cx - r - padding, cy - r - padding, cx + r + padding, cy + r + padding
            logger.debug(f"Circle bbox: {bbox}")
            return bbox
            
        elif elem.tag.endswith('ellipse'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))
            
            # Handle rotation
            transform = elem.get('transform', '')
            rotation_angle = 0
            if 'rotate' in transform:
                match = re.search(r'rotate\(([^)]+)\)', transform)
                if match:
                    rotation_angle = float(match.group(1))
            
            padding = 2.0
            rx += padding
            ry += padding
            
            if rotation_angle == 0:
                bbox = cx - rx, cy - ry, cx + rx, cy + ry
            else:
                # For rotated ellipses, calculate the bounding box of the rotated rectangle
                angle_rad = math.radians(rotation_angle)
                cos_angle = math.cos(angle_rad)
                sin_angle = math.sin(angle_rad)
                
                # Calculate the four corners of the rotated ellipse
                corners = [
                    (cx - rx, cy - ry),
                    (cx + rx, cy - ry),
                    (cx + rx, cy + ry),
                    (cx - rx, cy + ry)
                ]
                
                # Rotate each corner
                rotated_corners = []
                for px, py in corners:
                    new_x = (px - cx) * cos_angle - (py - cy) * sin_angle + cx
                    new_y = (px - cx) * sin_angle + (py - cy) * cos_angle + cy
                    rotated_corners.append((new_x, new_y))
                
                # Find the bounding box of rotated corners
                xs, ys = zip(*rotated_corners)
                bbox = min(xs), min(ys), max(xs), max(ys)
            
            logger.debug(f"Ellipse bbox: {bbox}")
            return bbox
            
        else:
            logger.warning(f"Unsupported element type: {elem.tag}")
            return 0, 0, 0, 0
            
    except Exception as e:
        logger.error(f"Error calculating bounding box for element {elem.tag}: {str(e)}")
        traceback.print_exc()
        return 0, 0, 0, 0

def calculate_group_bounding_box(group, elems_bbxs):
    """
    Calculate the bounding box for a group of elements using the precomputed bounding boxes.
    """
    # Extract all the bounding box coordinates
    bboxes = [elems_bbxs[elem] for elem in group]

    # Use zip to find min/max x and y values
    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[2] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)

    return min_x, min_y, max_x, max_y

def scale_translate_group(group, bboxes, canvas_width=512, canvas_height=512):
    min_x, min_y, max_x, max_y = calculate_group_bounding_box(group, bboxes)
    width = max_x - min_x
    height = max_y - min_y

    # Calculate scaling factors to fit within the canvas while maintaining aspect ratio
    scale_x = canvas_width / width
    scale_y = canvas_height / height
    scale = min(scale_x, scale_y)
    
    # Calculate centering offsets
    scaled_width = width * scale
    scaled_height = height * scale
    offset_x = (canvas_width - scaled_width) / 2
    offset_y = (canvas_height - scaled_height) / 2
    
    # Create a new SVG with the scaled and translated elements
    scaled_group = []
    for elem in group:
        if elem.tag.endswith('path'):
            # Convert element to string and parse with svgpathtools
            path_str = ET.tostring(elem, encoding='unicode')
            paths, attributes = svg2paths(path_str)
            
            if paths:
                # Scale and translate the path
                scaled_paths = []
                for path in paths:
                    # Scale and translate each segment
                    scaled_segments = []
                    for segment in path:
                        if isinstance(segment, Line):
                            start = segment.start * scale + (offset_x + offset_y * 1j)
                            end = segment.end * scale + (offset_x + offset_y * 1j)
                            scaled_segments.append(Line(start, end))
                        elif isinstance(segment, Arc):
                            # Scale radius and center
                            radius = segment.radius * scale
                            center = segment.center * scale + (offset_x + offset_y * 1j)
                            start = segment.start * scale + (offset_x + offset_y * 1j)
                            end = segment.end * scale + (offset_x + offset_y * 1j)
                            scaled_segments.append(Arc(start, radius, segment.rotation, 
                                                     segment.large_arc, segment.sweep, end))
                    
                    scaled_paths.append(Path(*scaled_segments))
                
                # Create new path element with scaled paths
                new_elem = ET.Element(elem.tag, elem.attrib)
                new_elem.set('d', ' '.join(str(p) for p in scaled_paths))
                scaled_group.append(new_elem)
                
        elif elem.tag.endswith('ellipse'):
            new_elem = ET.Element(elem.tag, elem.attrib)
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))
            
            new_elem.set('cx', str(((cx - min_x) * scale) + offset_x))
            new_elem.set('cy', str(((cy - min_y) * scale) + offset_y))
            new_elem.set('rx', str(rx * scale))
            new_elem.set('ry', str(ry * scale))
            scaled_group.append(new_elem)
            
        elif elem.tag.endswith('circle'):
            new_elem = ET.Element(elem.tag, elem.attrib)
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            
            new_elem.set('cx', str(((cx - min_x) * scale) + offset_x))
            new_elem.set('cy', str(((cy - min_y) * scale) + offset_y))
            new_elem.set('r', str(r * scale))
            scaled_group.append(new_elem)
        
        # Scale stroke width
        stroke_scale = elem.get("stroke-width")
        if stroke_scale:
            new_scale = str(float(stroke_scale) * scale)
        else:
            new_scale = str(scale)
        new_elem.set("stroke-width", new_scale)

    return scaled_group

def group_elements_by_proximity(elements, threshold=10):
    """
    Group elements based on their proximity and semantic ID.
    Elements are grouped if they overlap and have the same semantic ID.
    """
    groups = []
    grouped = [False] * len(elements)
    
    # Calculate bounding boxes for all elements first
    elems_bbxs = {}
    for elem in elements:
        try:
            bbox = calculate_bounding_box(elem)
            if bbox != (0, 0, 0, 0):  # Only include elements with valid bounding boxes
                elems_bbxs[elem] = bbox
        except Exception as e:
            logger.error(f"Error calculating bounding box: {str(e)}")
            continue
    
    if not elems_bbxs:
        logger.warning("No valid elements found for grouping")
        return [], {}
    
    for i, elem1 in enumerate(elements):
        if grouped[i] or elem1 not in elems_bbxs:
            continue

        # Get semantic ID of first element
        semantic_id1 = elem1.get('semanticId')
        if not semantic_id1:
            logger.warning(f"Element {i} has no semantic ID")
            continue

        group = [elem1]
        bx1, by1, bx2, by2 = elems_bbxs[elem1]
        
        for j, elem2 in enumerate(elements):
            if i == j or grouped[j] or elem2 not in elems_bbxs:
                continue

            # Check if semantic IDs match
            semantic_id2 = elem2.get('semanticId')
            if semantic_id2 != semantic_id1:
                continue

            bx3, by3, bx4, by4 = elems_bbxs[elem2]
            
            # Calculate the distance between bounding boxes
            # Check if boxes overlap or are very close
            overlap_x = min(bx2, bx4) - max(bx1, bx3)
            overlap_y = min(by2, by4) - max(by1, by3)
            
            # If boxes overlap or are within threshold distance
            if (overlap_x > -threshold and overlap_y > -threshold):
                group.append(elem2)
                grouped[j] = True

        if len(group) > 0:
            groups.append(group)
            logger.debug(f"Created group with {len(group)} elements of semantic ID {semantic_id1}")

    return groups, elems_bbxs

SVG_CATEGORIES = {
    1: "single door",
    2: "double door",
    3: "sliding door",
    4: "folding door",
    5: "revolving door",
    6: "rolling door",
    7: "window",
    8: "bay window",
    9: "blind window",
    10: "opening symbol",
    11: "sofa",
    12: "bed",
    13: "chair",
    14: "table",
    15: "TV cabinet",
    16: "Wardrobe",
    17: "cabinet",
    18: "gas stove",
    19: "sink",
    20: "refrigerator",
    21: "airconditioner",
    22: "bath",
    23: "bath tub",
    24: "washing machine",
    25: "squat toilet",
    26: "urinal",
    27: "toilet",
    28: "stairs",
    29: "elevator",
    30: "escalator",
    31: "row chairs",
    32: "parking spot",
    33: "wall",
    34: "curtain wall",
    35: "railing",
    36: "bg"
}