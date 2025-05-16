from parse_elements import *
import cairosvg
from lxml import etree as ET
import os
import argparse
from tqdm import tqdm

def extract_entities(root, ns):
    semantic_id_to_objects = {}
    for g in root.findall(ns + 'g'):
        for elem in g.findall("./*[@semanticId]"):
            if elem is not None:
                semantic_id = elem.get('semanticId')
                if semantic_id not in semantic_id_to_objects:
                    semantic_id_to_objects[semantic_id] = []
                semantic_id_to_objects[semantic_id].append(elem)
    
    return semantic_id_to_objects

def semantic2class(semanticId):
    semanticId = int(semanticId)
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
    for semanticId, elems in tqdm(semantic_id_to_objects.items()):
        category = semantic2class(semanticId)
        output_path = os.path.join(ds_path, split, "clip", category)
        os.makedirs(output_path, exist_ok=True)
        n_objects = len(os.listdir(output_path))

        groups, bboxes = group_elements_by_proximity(elems)
        for i, group in tqdm(enumerate(groups)):
            try:
                scaled_group = scale_translate_group(group, bboxes)
                filename = f"{n_objects+i+1:8}.jpg"
                file_path = os.path.join(output_path, filename)
                group2png(scaled_group, file_path)

            except Exception as e:
                print(f"Error at group of object {category}")
                continue
            print(f"Saved {category} object into {file_path}")



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
    for filename in tqdm(os.listdir(datapath)):
        file_path = os.path.join(datapath, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = root.tag[:-3]
        semantic_id_to_objects = extract_entities(root, ns)
        create_objects_imgs(semantic_id_to_objects, split, ds_path=dataset_path)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', type=str, default="./dataset",
                        help='dataset path'
                        )
    parser.add_argument('--split', type=str, default="train",
                        help='split to extract'
                        )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    extract_data(args.split, args.dataset_dir)

if __name__ == "__main__":
    main()