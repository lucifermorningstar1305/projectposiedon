from models import SiameseNetwork
import torch
import os

from transformers import ViTImageProcessor
from PIL import Image

if __name__ == "__main__":

    if not os.path.exists("./inference_model"):
        os.mkdir("inference_model")
    model = SiameseNetwork.load_from_checkpoint("./models/file/kaggle/working/checkpoints/siamese_net.ckpt-v1.ckpt").model.to("cpu")
    filepath = "./inference_model/similarity_model.onnx"
    input_sample = Image.open("./data/airplane.jpg")
    
    img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dummy_inputs = img_processor(input_sample, return_tensors="pt")
    print(dummy_inputs["pixel_values"].shape)

    # model.to_onnx(filepath, input_sample, export_params=True)
    torch.onnx.export(

        model, 
        tuple(dummy_inputs.values()),
        f = filepath,
        input_names = ["pixel_values"],
        output_names = ["img_representation"],
        dynamic_axes={"pixel_values": {0: "batch_size"},
                      "img_representation": {0: "batch_size"}},
        do_constant_folding=True,
        opset_version=13
    )

