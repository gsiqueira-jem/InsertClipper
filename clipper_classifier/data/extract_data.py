from parse_elements import *
import cairosvg
from lxml import etree as ET
import os
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging to only show errors
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a custom tqdm class that writes to stderr instead of stdout
class TqdmToLogger:
    def __init__(self, total, desc):
        self.tqdm = tqdm(total=total, desc=desc)
        
    def update(self, n=1):
        self.tqdm.update(n)
        
    def close(self):
        self.tqdm.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def extract_entities(root, ns):
    instance_id_to_objects = {}
    for g in root.findall(ns + 'g'):
        for elem in g.findall("./*[@semanticId]"):
            if elem is not None:
                instance_id = elem.get('instanceId')
                # Skip objects with no instance (instanceId="-1")
                if not instance_id or instance_id == "-1":
                    continue
                if instance_id not in instance_id_to_objects:
                    instance_id_to_objects[instance_id] = []
                instance_id_to_objects[instance_id].append(elem)
    
    return instance_id_to_objects

def semantic2class(semanticId):
    semanticId = int(semanticId)
    if semanticId in SVG_CATEGORIES:
        return SVG_CATEGORIES[semanticId]
    return "unknown"
    
def create_objects_imgs(instance_id_to_objects, split, ds_path="./dataset", original_filename=None):
    for instance_id, elems in instance_id_to_objects.items():
        if not elems:  # Skip if no elements
            continue
            
        # Get the semantic ID from the first element (all elements in group share same semantic ID)
        semantic_id = elems[0].get('semanticId')
        category = semantic2class(semantic_id)
        # Create folder using semantic ID
        output_path = os.path.join(ds_path, split, "clip", semantic_id)
        os.makedirs(output_path, exist_ok=True)

        # Calculate bounding boxes for the group
        bboxes = {}
        for elem in elems:
            bbox = calculate_bounding_box(elem)
            if bbox != (0, 0, 0, 0):  # Only include elements with valid bounding boxes
                bboxes[elem] = bbox

        if not bboxes:  # Skip if no valid bounding boxes
            continue

        try:
            scaled_group = scale_translate_group(elems, bboxes)
            # Use original filename and instance ID for the output filename
            base_filename = os.path.splitext(original_filename)[0] if original_filename else "unknown"
            filename = f"{base_filename}_{instance_id}.jpg"
            file_path = os.path.join(output_path, filename)
            group2png(scaled_group, file_path)
        except Exception as e:
            logger.error(f"Error processing object {category} (ID: {semantic_id}) with instance ID {instance_id}: {str(e)}")
            continue

def group2png(group, file_path):
    svg_str = "\n".join([ET.tostring(elem, encoding='unicode') for elem in group])
    svg_content = f"""
                    <svg xmlns='http://www.w3.org/2000/svg' width='512' height='512' viewBox='0 0 500 500'>
                        {svg_str}
                    </svg>
                    """
    
    cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=file_path)

def extract_data(split, dataset_path):
    datapath = f"{dataset_path}/{split}/{split}/svg_gt"
    files = os.listdir(datapath)
    
    # Use the custom tqdm class
    with TqdmToLogger(total=len(files), desc=f"Processing {split} files") as pbar:
        for filename in files:
            file_path = os.path.join(datapath, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                ns = root.tag[:-3]
                instance_id_to_objects = extract_entities(root, ns)
                create_objects_imgs(instance_id_to_objects, split, ds_path=dataset_path, original_filename=filename)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
            finally:
                pbar.update(1)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, default="./dataset",
                        help='dataset path'
                        )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"Extracting {split} data...")
        extract_data(split, args.dataset_dir)

if __name__ == "__main__":
    main()