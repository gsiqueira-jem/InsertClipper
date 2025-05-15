from shape import entity2shape
import os
import ezdxf
import clip 
from tqdm import tqdm
import ezdxf
import numpy as np
import cv2
from PIL import Image
import torch

def get_limits(shape_list):
    min_xs, max_xs, min_ys, max_ys = zip(*(shape.get_limits() for shape in shape_list))
    return min(min_xs), max(max_xs), min(min_ys), max(max_ys)

def draw_shapes(shape_list, image_size, offset, scale):
    img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    for shape in shape_list:
        shape.draw(img, offset, scale)
    
    return img
    

def render_block(doc, insert, image_size=(512,512), padding=20):
    try:
        name = insert.dxf.name
        block = doc.blocks.get(name)

        shape_list = [entity2shape(entity) for entity in block]
        shape_list = [shape for shape in shape_list if shape]
        
        min_x, max_x, min_y, max_y = get_limits(shape_list)
        
        dx = max_x - min_x
        dy = max_y - min_y
        
        scale = min((image_size[0] - 2 * padding) / dx, (image_size[1] - 2 * padding) / dy)
        offset = (min_x - padding / scale, min_y - padding / scale)

        img = draw_shapes(shape_list, image_size, offset, scale)
    
    except Exception as e:
        print(f"Error {e} rendering {block}")
        return None
    
    return img

def process_doc(doc):
    msp = doc.modelspace()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    exclude = []
    for insert in msp.query("INSERT"):
        img = render_block(doc, insert)
        if img is None:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        objects = ["a wall", "a window", "a door", "furtinure"]
        labels = [f"a CAD drawing of {rep} in a floor plan" for rep in objects]
        unknown_lb = "an unknown object in a CAD floor plan"
        
        labels.append(unknown_lb)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(labels).to(device)


        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            text_features = model.encode_text(text_tokens)

            logits_per_image, _ = model(img_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        arg_pred = np.argmax(probs)
        insert_id = insert.dxf.handle
        print(f"Predicted label for insert {insert_id}: {labels[arg_pred]} with a probability of {probs[arg_pred]}")
        if arg_pred >= 3:
            exclude.append(insert.dxf.handle)
    print(f"To exclude {exclude}")
    return exclude

def process_dxf_file(dxf_file):
    doc = ezdxf.readfile(dxf_file)
    return process_doc(doc)

def process_dxf_bytes(dxf_stream):
    doc = ezdxf.read(dxf_stream)
    return process_doc(doc)


def main():
    process_dxf_file("./dataset/1-Crosby_Original.dxf")


if __name__ == "__main__":
    main()