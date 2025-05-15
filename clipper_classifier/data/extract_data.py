import pandas as pd
import numpy as np
import cairosvg
from lxml import etree as ET
import os
import cv2

def extract_entities(root):
    semantic_id_to_objects = {}
    for g in root.findall(".//g"):
        for elem in g.findall(".//{*}semanticId"):
            if elem is not None:
                semantic_id = elem.text
                if semantic_id not in semantic_id_to_objects:
                    semantic_id_to_objects[semantic_id] = []
                semantic_id_to_objects[semantic_id].append(elem)
    
    return semantic_id_to_objects

def split_objects(img, output_dir, threshold_value=200):

    # Convert the image to grayscale for thresholding, but retain the original color image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask where foreground objects are detected
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find connected components (objects) in the thresholded binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create an output directory for individual object images
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each detected component (excluding the background label)
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    n_files = len(files)

    for label in range(1, num_labels):
        # Create a mask for each connected component (label)
        mask = (labels == label).astype(np.uint8) * 255

        # Apply the mask to the original image to extract the region corresponding to the object
        object_image = cv2.bitwise_and(img, img, mask=mask)

        # Get the bounding box for the component (for cropping)
        x, y, w, h, _ = stats[label]

        # Crop the region based on the bounding box
        cropped_object = object_image[y:y+h, x:x+w]

        # Save the cropped object as a separate image
        output_filename = os.path.join(output_dir, f"{n_files + label:06}.png")
        cv2.imwrite(output_filename, cropped_object)

    print(f"Objects have been saved in {output_dir}")


def semantic2class(semanticId):
    wall_cat = [33, 34]

    if semanticId <= 6:
        return "door"
    elif semanticId <= 10:
        return "window" 
    elif semanticId <= 27:
        return "furniture"
    elif semanticId <= 30 or (semanticId in wall_cat ):
        return "wall"
    else:
        return "object"
    
def create_objects_imgs(semantic_id_to_objects, split, ds_path="./dataset"):
    for semanticId, elems in semantic_id_to_objects.items():
        img = elements2png(semanticId, elems)
        
        category = semantic2class(semanticId)
        
        output_path = os.join(ds_path, split, category)
        img = elements2png(elems)
        
        split_objects(img, output_path)



def elements2png(elems):
    svg_str = "/n".join([ET.tostring(elem) for elem in elems])
    svg_content = f"<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='1000' viewBox='0 0 500 500'>{svg_str}</svg>"
    
    
    output = cairosvg.svg2png(bytestring=svg_content.encode("utf-8"))
    return output