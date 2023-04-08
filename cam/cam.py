import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms as T
import json
from PIL import Image
import numpy as np
import cv2
import requests


def load_image(url_or_path=""):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
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


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def _convert_to_array(img):
    img = np.array(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _convert_to_pil(img1)
    img2 = _convert_to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _convert_to_array(img_blended)


class CAMGenerator():
    def __init__(self, model):
        self.model = model
        self.feat_map = None # $f_{k}$
        self.register_hooks()

    def register_hooks(self):
        def forward_hook_fn(module, input, output):
            self.feat_map = input[0]

        for module in self.model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                module.register_forward_hook(forward_hook_fn)

    def _preprocess_class_activation_map(self, cam, h, w):
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(src=cam, dsize=(h, w))
        cam -= cam.min()
        cam *= 255 / cam.max()
        cam = np.clip(cam, 0, 255).astype("uint8")
        cam = _apply_jet_colormap(cam)
        return cam

    def get_class_activation_map(self, image, trg_class=None, with_image=False):
        if trg_class is None:
            class_scores = self.model(image)
            trg_class = torch.argmax(class_scores, dim=1).item()

        weights = model.fc.weight.data # $w^{c}_{k}$
        weighted_feat_maps = self.feat_map[0] * weights[trg_class][..., None, None]
        cam = weighted_feat_maps.sum(dim=0)

        _, _, h, w = image.shape
        cam = self._preprocess_class_activation_map(cam=cam, h=h, w=w)
        if with_image:
            img = tensor_to_array(image)
            result = _blend_two_images(img1=img, img2=cam, alpha=0.7)
        else:
            result = cam
        return result


def save_image(img, path):
    _convert_to_pil(img).save(str(path))


if __name__ == "__main__":
    idx2class = json.load(open("/Users/jongbeomkim/Downloads/imagenet_class_index.json"))

    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.eval()
    model.zero_grad()

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224), antialias=True),
            T.CenterCrop((224)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    img = load_image(
        "/Users/jongbeomkim/Downloads/32653304805_5e6a0544b7_c.jpg"
    )
    image = transform(img).unsqueeze(0)
    cam_gen = CAMGenerator(model)
    cam = cam_gen.get_class_activation_map(image=image, with_image=True)
    save_image(img=cam, path="/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/samples/deer.jpg")