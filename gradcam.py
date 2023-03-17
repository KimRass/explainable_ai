# References
    # https://github.com/jacobgil/pytorch-grad-cam


import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchviz import make_dot
import cv2
from PIL import Image


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()
trg_layer = model.layer4[-1]
trg_layer.register_forward_hook

img = load_image("/Users/jongbeomkim/Downloads/download.png")
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
image = transform(img).unsqueeze(0)
image.shape

# make_dot(model(image), params=dict(model.named_parameters())).render("/Users/jongbeomkim/Downloads/resnet50", format="jpg")

output = model(image) # yc
output[0]


# class ClassifierOutputTarget:
#     def __init__(self, category):
#         self.category = category

#     def __call__(self, model_output):
#         if len(model_output.shape) == 1:
#             return model_output[self.category]
#         return model_output[:, self.category]

# ClassifierOutputTarget(281)