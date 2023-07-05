import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision.transforms as T
import json
import numpy as np
import cv2
from pathlib import Path

from image_utils import (
    load_image,
    _blend_two_images,
    save_image,
    _apply_jet_colormap,
    _reverse_jet_colormap,
    draw_bboxes
)

# IDX2CLASS = json.load(open("./imagenet_class_index.json"))
IDX2CLASS = json.load(open("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/imagenet_class_index.json"))


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


class ClassActivationMapper():
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


if __name__ == "__main__":
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.eval()
    model.zero_grad()

    cam_gen = ClassActivationMapper(model)

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

    dir = Path("/Users/jongbeomkim/Desktop/workspace/explainable_ai/cam/examples")
    for img_path in dir.glob("*.jpg"):
        if "_original." not in img_path.name:
            continue

        img = load_image(img_path)
        image = transform(img).unsqueeze(0)

        cam, category = cam_gen.get_class_activation_map(image=image, with_image=True)
        print(IDX2CLASS[str(category)][1])
        save_image(img=cam, path=dir/f"""{img_path.stem.rsplit("_", 1)[0]}_cam.png""")

        # cam_gen = ClassActivationMapper(model)
        bboxes = cam_gen.get_top_n_bboxes(image=image, n=5, thresh=0.7)
        drawn = draw_bboxes(img=tensor_to_array(image), bboxes=bboxes[0: 1, ...,])
        save_image(img=drawn, path=dir/f"""{img_path.stem.rsplit("_", 1)[0]}_bboxes.png""")
