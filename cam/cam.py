import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import requests
from pathlib import Path


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
    blended_img = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _convert_to_array(blended_img)


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
        cam = cv2.resize(src=cam, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
        cam -= cam.min()
        cam *= 255 / cam.max()
        cam = np.clip(cam, 0, 255).astype("uint8")
        cam = _apply_jet_colormap(cam)
        return cam

    def get_class_activation_map(self, image, category=None, rank=1, with_image=False):
        if category is None:
            class_scores = self.model(image)
            # category = torch.argmax(class_scores, dim=1).item()
            category = torch.argsort(class_scores, dim=1, descending=True)[..., rank - 1].item()

        weights = model.fc.weight.data # $w^{c}_{k}$
        weighted_feat_maps = self.feat_map[0] * weights[category][..., None, None]
        cam = weighted_feat_maps.sum(dim=0)

        _, _, h, w = image.shape
        cam = self._preprocess_class_activation_map(cam=cam, h=h, w=w)
        if with_image:
            img = tensor_to_array(image)
            result = _blend_two_images(img1=img, img2=cam, alpha=0.7)
        else:
            result = cam
        return result, category

    def get_top_n_bboxes(self, image, n=1, thresh=0.2):
        def _cam_to_bboxes(cam):
            cam = _reverse_jet_colormap(cam)
            _, obj_mask = cv2.threshold(src=cam, thresh=int(255 * (1 - thresh)), maxval=255, type=cv2.THRESH_BINARY)
            _, _, stats, _ = cv2.connectedComponentsWithStats(image=obj_mask)
            sorted_stats = stats[1:, ...][np.argsort(stats[1:, cv2.CC_STAT_AREA])[:: -1]]
            x1, y1, w, h, _ = sorted_stats[0, ...]
            return x1, y1, x1 + w, y1 + h

        bboxes = list()
        for rank in range(1, n + 1):
            cam, category = cam_gen.get_class_activation_map(image=image, rank=rank, with_image=False)
            x1, y1, x2, y2 = _cam_to_bboxes(cam)
            bboxes.append((x1, y1, x2, y2, category))
        bboxes = np.array(bboxes)
        return bboxes


def save_image(img, path):
    _convert_to_pil(img).save(str(path))


def _reverse_jet_colormap(img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = list(map(tuple, _apply_jet_colormap(gray_values).reshape(256, 3)))
    color_to_gray_map = dict(zip(color_values, gray_values))

    out = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], axis=2, arr=img)
    return out


def draw_bboxes(img, bboxes):
    canvas = _convert_to_pil(img)
    draw = ImageDraw.Draw(canvas)

    for x1, y1, x2, y2, label in bboxes:
        draw.rectangle(
            xy=(x1, y1, x2, y2),
            outline=(255, 0, 0),
            fill=None,
            width=max(1, int(min(x2 - x1, y2 - y1) * 0.02))
        )

    for x1, y1, x2, y2, label in bboxes:
        draw.text(
            xy=(x1, y1 - 4),
            text=" " + idx2class[str(label)][1],
            fill="white",
            stroke_fill="black",
            stroke_width=2,
            font=ImageFont.truetype(
                # font="./fonts/Pretendard-Regular.otf",
                font="/Users/jongbeomkim/Downloads/Pretendard-1.3.6/public/static/Pretendard-Regular.otf",
                size=max(10, int(min(40, min(x2 - x1, y2 - y1) * 0.12)))
            ),
            anchor="la"
        )
    return canvas


if __name__ == "__main__":
    # idx2class = json.load(open("./imagenet_class_index.json"))
    idx2class = json.load(open("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/imagenet_class_index.json"))

    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.eval()
    model.zero_grad()

    cam_gen = CAMGenerator(model)

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

    dir = Path("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/samples")
    for img_path in dir.glob("*"):
        if "_original." not in img_path.name:
            continue

        img = load_image(img_path)
        image = transform(img).unsqueeze(0)

        cam, category = cam_gen.get_class_activation_map(image=image, with_image=True)
        print(idx2class[str(category)][1])
        save_image(img=cam, path=dir/f"""{img_path.stem.rsplit("_", 1)[0]}_cam.png""")

        # cam_gen = CAMGenerator(model)
        bboxes = cam_gen.get_top_n_bboxes(image=image, n=5, thresh=0.7)
        drawn = draw_bboxes(img=tensor_to_array(image), bboxes=bboxes[0: 1, ...,])
        save_image(img=drawn, path=dir/f"""{img_path.stem.rsplit("_", 1)[0]}_bboxes.png""")
