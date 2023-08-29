from typing import List, Any, Optional, Dict, Tuple

import gradio as gr
import numpy as np
import torch
import lightning.pytorch as pl
import transformers
import os

from PIL import Image, ImageDraw
from models import SiameseNetwork
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob


def get_image_embeddings(image: Image.Image | List[Image.Image]) -> np.ndarray:
    """
    Function to return the image embeddings of an image.

    :param image: an RGB image of shape (H x W x C)

    :returns: the embeddings of the image.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork.load_from_checkpoint("./models/file/kaggle/working/checkpoints/siamese_net.ckpt-v1.ckpt").to(device)
    model = model.model
    preprocessor = transformers.ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    model.eval()
    with torch.no_grad():
        img_inputs = preprocessor(images=image, return_tensors="pt").to(device)
        embeddings = model(**img_inputs).pooler_output.detach().cpu().numpy()

    return embeddings

def reference_img_upload(img_input: Image, label_input: str):
    """
    Function to save the reference images in your local server.

    :param img_input: an RGB image of the format (H x W x C)
    :param label_input: the label for the image.

    :returns: None
    """

    img = img_input
    img.save(os.path.join("./reference_imgs", f"{label_input}.jpg"))


def img_classify(input_img: Image) -> str:

    """
    Function to classify images based on image similarity

    :param input: an RGB image of shape (H x W x C)

    :returns: the label of the image
    """
    reference_imgs = glob("./reference_imgs/*.jpg")

    labels = [img_path.split("/")[-1].split(".")[0] for img_path in reference_imgs]

    input_img_embedding = get_image_embeddings(input_img.resize((224, 224), resample=Image.Resampling.BILINEAR))
    
    ref_imgs = list()
    
    for img in reference_imgs:
        img = Image.open(img)
        ref_imgs.append(img.convert("RGB").resize((224, 224), resample=Image.Resampling.BILINEAR))

    
    img_embeddings = get_image_embeddings(ref_imgs)
    dist = np.sqrt(np.power(input_img_embedding - img_embeddings, 2)).sum(axis=1)
    print(cosine_similarity(input_img_embedding, img_embeddings))
    
    print(dist)

    idx = np.argmin(dist)
    i = len(glob(f"./dumps/{labels[idx]}*"))
    fn = f"{labels[idx]}.jpg" if i == 0 else f"{labels[idx]}_{i}.jpg"
    input_img.save(os.path.join("./dumps", fn))
    return labels[idx]
    
def zero_shot_object_detection(query_image: Image.Image, target_image: Image.Image, ) -> Image.Image:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    inputs = img_processor(images=target_image, query_images=query_image, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
        target_sizes = torch.tensor([target_image.size[::-1]])
        results = img_processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(target_image)
    scores = results["scores"].tolist()
    boxes = results["boxes"].tolist()

    for box, score in zip(boxes, scores):
        xmin, ymin, xmax, ymax = box
        print(score)
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=14)

    return target_image


upload_reference_image = gr.Interface(reference_img_upload, 
                                    inputs=[gr.Image(type="pil", label="Reference Image"), 
                                            gr.Text(value="", placeholder="class", label="Label of the image")],
                                    outputs = None)

classify_images = gr.Interface(img_classify, 
                               inputs=gr.Image(type="pil", label="Image", interactive=True),
                               outputs=gr.Text(value=""), cache_examples=True)

track_object = gr.Interface(zero_shot_object_detection, 
                            inputs = [gr.Image(type="pil", label="Query Image", interactive=True),
                                      gr.Image(type="pil", label="Target Image", interactive=True)],
                            outputs = gr.Image(type="pil", label="Detected Objects", interactive=False))

demo = gr.TabbedInterface([upload_reference_image, classify_images, track_object], 
                          tab_names=["Upload Reference Images", "Classify Images", "Object Detection"],
                          title="Project Poseidon 🌊🔱")


if not os.path.exists("./reference_imgs/"):
    os.mkdir("./reference_imgs")


if not os.path.exists("./dumps"):
    os.mkdir("./dumps")


if __name__ == "__main__":
    
    demo.launch()








