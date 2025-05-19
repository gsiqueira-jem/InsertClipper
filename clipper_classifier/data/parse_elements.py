import re
import math
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.ERROR)  # Changed to ERROR level
logger = logging.getLogger(__name__)

def parse_simple_path(path_str):
    tokens = path_str.strip().split()
    i = 0
    segments = []

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd == 'M':
            x, y = map(float, tokens[i].split(','))
            segments.append(('M', x, y))
            i += 1

        elif cmd == 'L':
            x, y = map(float, tokens[i].split(','))
            segments.append(('L', x, y))
            i += 1

        elif cmd == 'A':
            rx, ry = map(float, tokens[i].split(','))
            i += 1
            rotation = float(tokens[i])
            i += 1
            laf, sf = map(int, tokens[i].split(','))
            i += 1
            x, y = map(float, tokens[i].split(','))
            i += 1
            segments.append(('A', rx, ry, rotation, laf, sf, x, y))

        else:
            raise ValueError(f"Unexpected command: {cmd}")

    return segments


def calculate_bounding_box(elem):
    """
    Calculate the bounding box of an SVG element (including paths).
    """
    try:
        padding = 2.0
        if elem.tag.endswith('path'):
            path_data = elem.get('d', '')
            coordinates = parse_simple_path(path_data)
            if not coordinates:
                logger.error(f"No coordinates found in path: {path_data[:50]}...")
                return 0, 0, 0, 0

            # Calculate the bounding box
            xs = []
            ys = []
            for point in coordinates:
                if point[0] == 'M' or point[0] == 'L':
                    xs.append(point[1])
                    ys.append(point[2])
                elif point[0] == 'A':
                    xs.append(point[6] + point[1])
                    xs.append(point[6] - point[1])
                    ys.append(point[7] + point[2])
                    ys.append(point[7] - point[2])
            
            if not xs or not ys:
                logger.error(f"Empty coordinates after extraction for path")
                return 0, 0, 0, 0
                
            bbox = min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding
            return bbox
        
        elif elem.tag.endswith('circle'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            bbox = cx - r - padding, cy - r - padding, cx + r + padding, cy + r + padding
            return bbox
        
        elif elem.tag.endswith('ellipse'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))
            
            # Handle rotation
            pivot_x = cx
            pivot_y = cy
            angle_deg = 0

            transform = elem.get('transform', '')
            if transform and transform.startswith("rotate"):
                parts = transform.strip("rotate()").split(',')
                angle_deg = float(parts[0])
                if len(parts) == 3:
                    pivot_x = float(parts[1])
                    pivot_y = float(parts[2])
            
            # Rotate center around pivot
            theta_rad = math.radians(angle_deg)
            dx = cx - pivot_x
            dy = cy - pivot_y
            rotated_cx = pivot_x + (dx * math.cos(theta_rad) - dy * math.sin(theta_rad))
            rotated_cy = pivot_y + (dx * math.sin(theta_rad) + dy * math.cos(theta_rad))
            
            # Project ellipse radii onto rotated coordinate system
            cos_angle = math.cos(theta_rad)
            sin_angle = math.sin(theta_rad)

            # Calculate bounding box
            width = abs(rx * cos_angle) + abs(ry * sin_angle)
            height = abs(rx * sin_angle) + abs(ry * cos_angle)

            left = rotated_cx - width - padding
            right = rotated_cx + width + padding
            top = rotated_cy - height - padding
            bottom = rotated_cy + height + padding

            return left, top, right, bottom

            
        else:
            logger.error(f"Unsupported element type: {elem.tag}")
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

    # Calculate scaling factors to fit within the 512x512 canvas while maintaining aspect ratio
    scale_x = canvas_width / width
    scale_y = canvas_height / height
    scale = min(scale_x, scale_y)  # We take the smaller scaling factor to maintain aspect ratio
    
    for elem in group:
        if elem.tag.endswith('ellipse'):
            # For <ellipse> elements: scale and translate
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            rx = float(elem.get('rx', 0))
            ry = float(elem.get('ry', 0))

            elem.set('cx', str((cx - min_x) * scale_x))
            elem.set('cy', str((cy - min_y) * scale_y ))
            elem.set('rx', str(rx * scale_x))
            elem.set('ry', str(ry * scale_y))

            transform = elem.get('transform', '')
            if 'rotate' in transform:
                angle, px, py = map(float, transform.strip("rotate()").split(','))
                scaled_px = (px - min_x) * scale_x
                scaled_py = (py - min_y) * scale_y
                new_transform = f"rotate({angle}, {scaled_px}, {scaled_py})"
                elem.set('transform', new_transform)

                    
        elif elem.tag.endswith('circle'):
            # For <circle> elements: scale and translate
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))

            elem.set('cx', str((cx - min_x) * scale_x))
            elem.set('cy', str((cy - min_y) * scale_y))
            elem.set('r', str(r * (scale_x + scale_y)/2))
        
        elif elem.tag.endswith('path'):
            # For <path> elements: scale and translate (handle coordinates)
            path_data = elem.get('d', '')
            coords = parse_simple_path(path_data)
    
            # Scale and translate the coordinates
            scaled_coords = []
            for point in coords:
                cmd = point[0]
                if cmd == 'M' or cmd == 'L':
                    x, y = point[1], point[2]
                    new_x = (x - min_x) * scale_x 
                    new_y = (y - min_y) * scale_y 
                    scaled_coords.append((cmd, new_x, new_y))
                elif cmd == 'A':
                    rx, ry = point[1], point[2]
                    rotation_deg = point[3]
                    laf, sf = point[4], point[5]
                    end_x, end_y = point[6], point[7]

                    new_rx = rx * scale_x
                    new_ry = ry * scale_y 
                    new_x = (end_x - min_x) * scale_x 
                    new_y = (end_y - min_y) * scale_y 

                    theta_rad = math.radians(rotation_deg)
                    rotated = math.degrees(math.atan2(math.sin(theta_rad) * scale_x, math.cos(theta_rad) * scale_y)) % 360
                    
                    scaled_coords.append((cmd, new_rx, new_ry, rotated, laf, sf, new_x, new_y))
            
            # Rebuild the path data string with the scaled coordinates
            scaled_path_data = ''
            for point in scaled_coords:
                cmd = point[0]
                if cmd == 'M' or cmd == 'L':
                    x, y = point[1], point[2]
                    scaled_path_data += f"{cmd} {x},{y} "
                elif cmd == 'A': 
                    rx, ry, rotation_deg, laf, sf, end_x, end_y = point[1:]
                    scaled_path_data += f"{cmd} {rx},{ry} {rotation_deg} {laf},{sf} {end_x},{end_y} "
            elem.set('d', scaled_path_data)

        stroke_scale = elem.get("stroke-width")
        if stroke_scale:
            new_scale = str(float(stroke_scale) * 50)
        else:
            new_scale = "5"
        elem.set("stroke-width", new_scale)

    return group

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