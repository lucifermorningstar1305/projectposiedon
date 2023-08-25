from typing import Tuple, Dict, List, Any, Callable, Optional

import numpy as np
import streamlit as st
import torch
import lightning.pytorch as pl
import os
import sys

from PIL import Image
from transformers import ViTImageProcessor
from models import SiameseNetwork

@st.cache_resource
def get_dists(anchor_img: Image, pos_img: Image, neg_img: Image) -> Tuple[np.ndarray, np.ndarray]:
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork.load_from_checkpoint("./models/file/kaggle/working/checkpoints/siamese_net.ckpt-v1.ckpt").to(device)
    model.eval()
    with torch.no_grad():
        anchor_img_feats, pos_img_feats, neg_img_feats = model(processor(images=anchor_img, return_tensors="pt").to(device),
                                                               processor(images=pos_img, return_tensors="pt").to(device),
                                                               processor(images=neg_img, return_tensors="pt").to(device))
        
    
    d_pos = torch.sqrt((anchor_img_feats - pos_img_feats).pow(2).sum(1)).detach().cpu().numpy()
    d_neg = torch.sqrt((anchor_img_feats - neg_img_feats).pow(2).sum(1)).detach().cpu().numpy()

    return d_pos, d_neg


if __name__ == "__main__":

    st.header("Project Poseidon 🔱🌊")

    st.markdown("Welcome to **Project Poseidon 🔱🌊**.")
    INTRO = """In this framework you will need to upload two reference images for each of the Two Types and also 
     add the labels for each of these two images. Once done you can then upload the image that you want to which
     category it belongs to. 
     Best part of this project: This can work on any kind of data :wink:"""
    st.markdown(INTRO)
    pos_img, neg_img, anchor_img = None, None, None
    st.subheader("Type-1 Image")
    pos_uploaded_img = st.file_uploader("Choose an POS image", type=["png", "jpeg"])
    pos_label = st.text_input("Type 1 Label: ", value="")

    st.subheader("Type-2 Image")
    neg_uploaded_img = st.file_uploader("Choose an NEG image", type=["png", "jpeg"])
    neg_label = st.text_input("Type 2 Label: ", value="")

    ref_uploaded_img = st.file_uploader("Upload an image", type=["png", "jpeg"])

    if pos_uploaded_img is not None:
        pos_img_bytes = pos_uploaded_img.getvalue()
        pos_img = Image.open(pos_uploaded_img).resize((224, 224), resample=Image.Resampling.BILINEAR)

    if neg_uploaded_img is not None:
        neg_img_bytes = neg_uploaded_img.getvalue()
        neg_img = Image.open(neg_uploaded_img).resize((224, 224), resample=Image.Resampling.BILINEAR)

    if ref_uploaded_img is not None:
        ref_img_bytes = ref_uploaded_img.getvalue()
        ref_img = Image.open(ref_uploaded_img).resize((224, 224), resample=Image.Resampling.BILINEAR)

        if pos_uploaded_img is None or neg_uploaded_img is None or ref_uploaded_img is None:
            raise Exception("Missing one of the 3 images. Please re-upload all 3 images")
    
        if len(pos_label) == 0 or len(neg_label) == 0:
            raise Exception("Missing label for one of the two types. Please re-enter the labels for the two types.")
        

        col1, col2= st.columns(2)
        with col1:
            st.header(pos_label)
            st.image(pos_img_bytes)
            

        with col2:
            st.header(neg_label)
            st.image(neg_img_bytes)
            

        st.header("Image to be Classified")
        st.image(ref_img_bytes)
        with st.spinner("Processing ...."):
            d_pos, d_neg = get_dists(ref_img, pos_img, neg_img)

        print(d_pos, d_neg, 1/(1 + d_pos), 1/(1 + d_neg))
        if d_pos < d_neg:
            st.info(f"It is a **{pos_label}**")

        else:
            st.info(f"It is a **{neg_label}**")

        
