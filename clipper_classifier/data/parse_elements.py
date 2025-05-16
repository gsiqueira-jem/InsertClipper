import re
import math

def extract_coordinates_from_path(path_data):
    pattern = r"([MLA])\s*([+-]?\d*\.\d+|\d+),([+-]?\d*\.\d+|\d+)"
    coords = re.findall(pattern, path_data)
    return [(cmd, float(x), float(y)) for cmd, x, y in coords]

def calculate_bounding_box(elem):
    """
    Calculate the bounding box of an SVG element (including paths).
    """
    if elem.tag.endswith('path'):
        # For <path> elements: extract coordinates from the `d` attribute
        path_data = elem.get('d', '')
        coordinates = extract_coordinates_from_path(path_data)
        if not coordinates:
            return 0, 0, 0, 0  # If no coordinates are found, return a default bounding box

        # Calculate the bounding box by finding min/max x and y values from coordinates
        _, xs, ys = zip(*coordinates)
        return min(xs), min(ys), max(xs), max(ys)
    
    elif elem.tag.endswith('circle'):
        # For <circle> elements: cx, cy, r (radius)
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        r = float(elem.get('r', 0))
        return cx - r, cy - r, cx + r, cy + r
    
    elif elem.tag.endswith('ellipse'):
        # For <ellipse> elements: cx, cy, rx, ry, rotation (transform attribute)
        cx = float(elem.get('cx', 0))
        cy = float(elem.get('cy', 0))
        rx = float(elem.get('rx', 0))
        ry = float(elem.get('ry', 0))
        transform = elem.get('transform', '')

        # Check for rotation in the transform attribute (e.g., "rotate(45)")
        rotation_angle = 0
        if 'rotate' in transform:
            match = re.search(r'rotate\(([^)]+)\)', transform)
            if match:
                rotation_angle = float(match.group(1))

        # Convert the rotation angle to radians
        angle_rad = math.radians(rotation_angle)

        # The bounding box of a rotated ellipse
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # Compute the bounding box corners (ellipse vertices)
        # Rotate the four extreme points (top, bottom, left, right) of the ellipse
        points = [
            (cx - rx, cy),  # Left
            (cx + rx, cy),  # Right
            (cx, cy - ry),  # Top
            (cx, cy + ry),  # Bottom
        ]

        # Rotate each point
        rotated_points = []
        for px, py in points:
            new_x = (px - cx) * cos_angle - (py - cy) * sin_angle + cx
            new_y = (px - cx) * sin_angle + (py - cy) * cos_angle + cy
            rotated_points.append((new_x, new_y))

        # Find min/max x and y values to form the bounding box
        xs, ys = zip(*rotated_points)
        return min(xs), min(ys), max(xs), max(ys)



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
            coords = extract_coordinates_from_path(path_data)
    
            # Scale and translate the coordinates
            scaled_coords = []
            for cmd, x, y in coords:
                new_x = (x - min_x) * scale_x 
                new_y = (y - min_y) * scale_y 
                scaled_coords.append((cmd, new_x, new_y))
            
            # Rebuild the path data string with the scaled coordinates
            scaled_path_data = ' '.join(f"{cmd} {x},{y}" for cmd, x, y in scaled_coords)
            path_data = elem.set('d', scaled_path_data)

        stroke_scale = elem.get("stroke-width")
        if stroke_scale:
            new_scale = str(float(stroke_scale) * 50)
        else:
            new_scale = "5"
        elem.set("stroke-width", new_scale)

    return group



def group_elements_by_proximity(elements, threshold=10):
    """
    Group elements based on their proximity using their bounding boxes.
    """
    groups = []
    grouped = [False] * len(elements)  # Keep track of which elements have been grouped
    
    elems_bbxs = {elem : calculate_bounding_box(elem) for elem in elements}
    
    for i, elem1 in enumerate(elements):
        if grouped[i]:
            continue

        group = [elem1]
        bx1, by1, bx2, by2 = elems_bbxs[elem1]
        for j, elem2 in enumerate(elements):
            if i == j or grouped[j]:
                continue

            bx3, by3, bx4, by4 = elems_bbxs[elem2]
            # Check if bounding boxes overlap or are close enough (based on threshold)
            if (abs(bx1 - bx3) < threshold and abs(by1 - by3) < threshold):
                group.append(elem2)
                grouped[j] = True

        groups.append(group)

    return groups, elems_bbxs