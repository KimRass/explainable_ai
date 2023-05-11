import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
import cv2
from typing import Literal
from PIL import Image
import requests

from process_images import (
    load_image,
    show_image,
    save_image,
    resize_image,
    get_width_and_height,
    _blend_two_images,
    _apply_jet_colormap,
    _rgba_to_rgb
)


def denormalize_array(img):
    copied_img = img.copy()
    copied_img -= copied_img.min()
    copied_img /= copied_img.max()
    copied_img *= 255
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze()
    if copied_tensor.ndim == 3:
        copied_tensor = copied_tensor.permute((1, 2, 0)).detach().cpu().numpy()
    elif copied_tensor.ndim == 2:
        copied_tensor = copied_tensor.detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def print_all_layers(model):
    print(f"""|         NAME         |                            MODULE                            |""")
    print(f"""|----------------------|--------------------------------------------------------------|""")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and name:
            print(f"""| {name:20s} | {str(type(module)):60s} |""")


def _get_target_layer(layer):
    return eval(
        "model" + "".join(
            [f"""[{i}]""" if i.isdigit() else f""".{i}""" for i in layer.split(".")]
        )
    )


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_map = None

    def get_feature_map(self, image, layer):
        def forward_hook_fn(module, input, output):
            self.feat_map = output

        trg_layer = _get_target_layer(layer)
        trg_layer.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_map


def convert_feature_map_to_attention_map(feat_map, img, mode=Literal["bw", "jet"], p=1):
    feat_map = feat_map.sum(axis=1)
    feat_map = feat_map ** p

    feat_map = convert_tensor_to_array(feat_map)
    w, h = get_width_and_height(img)
    feat_map = resize_image(img=feat_map, w=w, h=h)
    if mode == "bw":
        output = np.concatenate([img, feat_map[..., None]], axis=2)
    elif mode == "jet":
        feat_map = _apply_jet_colormap(feat_map)
        output = _blend_two_images(img1=img, img2=feat_map, alpha=0.6)
    output = _rgba_to_rgb(output)
    return output


def sort_feature_map(feat_map):
    argsort = torch.argsort(feat_map.sum(dim=(2, 3)), dim=1, descending=True)[0]
    feat_map = feat_map[:, argsort]
    return feat_map


if __name__ == "__main__":
    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    # print_all_layers(model)

    img = load_image("https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-147786673.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((227, 227)),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    )
    image = transform(img).unsqueeze(0)

    square_size = 227 // 4
    extractor = FeatureMapExtractor(model)
    canvas = np.zeros((18, 18), dtype="float")
    for i in range(0, 227 + 1 - square_size, 10):
        for j in range(0, 227 + 1 - square_size, 10):
            occluded_image = image.clone()
            occluded_image[:, :, i: i + square_size, j: j + square_size] = image.mean(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)

            feat_map1 = extractor.get_feature_map(image=occluded_image, layer="features.5")
            # feat_map2 = extractor.get_feature_map(image=image, layer="features.5")
            # feat_map3 = extractor.get_feature_map(image=image, layer="features.12")

            feat_map1 = sort_feature_map(feat_map1)
            strongest_feat_map = feat_map1[:, 0, ...]
            canvas[i // 10, j // 10] = strongest_feat_map.sum().item()

            # print(i, j, strongest_feat_map.sum().item())


    canvas -= canvas.min()
    canvas /= canvas.max()
    canvas *= 255
    canvas = canvas.astype("uint8")
    temp = _apply_jet_colormap(canvas)
    show_image(temp)