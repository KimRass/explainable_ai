import torch
import torch.nn as nn
import torchvision
from torchvision.models import GoogLeNet, GoogLeNet_Weights
import torchvision.transforms as T
import json
from PIL import Image
import numpy as np

idx2class = json.load(open("/Users/jongbeomkim/Downloads/imagenet_class_index.json"))

# model = GoogLeNet(weights=GoogLeNet_Weights.DEFAULT)
model = torch.hub.load("pytorch/vision:v0.10.0", model="googlenet", pretrained=True)

model

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((224)),
        T.CenterCrop((224)),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
img = load_image("https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-147786673.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*")
show_image(img)

image = transform(img)

class ClassActivationMap():
    def __init__(self, model):
        self.model = model
        self.feat_map = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook_fn(module, input, output):
            self.feat_map = input[0]

        for module in self.model.modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                module.register_forward_hook(forward_hook_fn)

    def get_gap_feature_map(self, x):
        probs = self.model(x)
        pred_label = torch.argmax(probs, dim=1).item()

        self.weight = model.fc.weight.data
        return self.feat_map


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


def batched_image_to_grid(image, normalize=False, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    b, _, h, w = image.shape
    pad = max(2, int(max(h, w) * 0.04))
    n_rows = int(b ** 0.5)
    grid = torchvision.utils.make_grid(tensor=image, nrow=n_rows, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    if normalize:
        grid *= variance
        grid += mean
    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    if n_rows > 1:
        for k in range(n_rows + 1):
            grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
            grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    return grid


cam = ClassActivationMap(model)
feat_map = cam.get_gap_feature_map(image.unsqueeze(0))

weight = model.fc.weight.data
feat_map[0].shape, weight[pred_label].unsqueeze(0).shape
torch.matmul(weight[pred_label].unsqueeze(0), feat_map[0])

feat_map.shape
model

weight.T.shape
matmul = torch.matmul(feat_map.squeeze(), weight.T)
matmul[pred_label]
probs = model(image.unsqueeze(0))
probs.shape
probs.max(dim=1)
pred_label = torch.argmax(probs, dim=1).item()

idx2class[str(pred_label)]
idx2class

grid = batched_image_to_grid(image=image.unsqueeze(0), normalize=True)
show_image(grid)