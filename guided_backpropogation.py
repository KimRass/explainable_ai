# References
    # https://leslietj.github.io/2020/07/22/Deep-Learning-Guided-BackPropagation/

import json
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torchviz import make_dot
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from process_images import (
    load_image
)

idx2class = json.load(open("/Users/jongbeomkim/Downloads/imagenet_class_index.json"))

model.eval()



# make_dot(model(image), params=dict(model.named_parameters())).render("/Users/jongbeomkim/Downloads/resnet50", format="jpg")
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

def normalize(x):
    norm = (x - x.mean()) / x.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm = norm.clip(0, 1)
    return norm


class GuidedBackpropogation():
    def __init__(self, model):
        self.model = model
        self.feat_maps = list() # Stores $f^{1}, f^{2}, ...$
        self.reconstructed_image = None # store R0
        self.register_hooks()

    def register_hooks(self):
        def first_layer_backward_hook_fn(module, grad_in, grad_out):
            # print(grad_out)
            # self.reconstructed_image = grad_in
            self.reconstructed_image = grad_in[0]

        def forward_hook_fn(module, input, output):
            self.feat_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            # print(self.feat_maps)
            feat_map = self.feat_maps.pop() # $f^{l}_{i}$

            # $R^{l}_{i} = (f^{l}_{i} > 0) \cdot (R^{l + 1}_{i} > 0) \cdot R^{l + 1}_{i}$
            new_grad_in = (feat_map > 0) * (grad_out[0] > 0) * grad_out[0]
            return (new_grad_in,)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                print(module)
                # The `hook` will be called every time after `forward()` has computed an output.
                # `register_forward_hook(module, input, output)`:
                module.register_forward_hook(forward_hook_fn)
                # `register_backward_hook(module, grad_in, grad_out)`:
                    # `grad_out`: 현재 레이어의 출력에 대한 모델 출력의 기울기
                module.register_backward_hook(backward_hook_fn)

        first_layer = list(self.model.modules())[1]
        first_layer.register_backward_hook(first_layer_backward_hook_fn)

    def visualize(self, x):
        output = model(x) # yc

        self.model.zero_grad()

        pred = output[0].detach().cpu().numpy()
        pred_class = np.argmax(pred)

        grad_trg_map = torch.zeros(output.shape, dtype=torch.float)
        grad_trg_map[0, pred_class] = 1

        output.backward(grad_trg_map)
        return self.reconstructed_image.data[0]
image = transform(img).unsqueeze(0).requires_grad_()
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model
guided_backprop = GuidedBackpropogation(model)
# guided_backprop.register_hooks()
result = guided_backprop.visualize(image)


# temp = np.clip(a=(result.detach().cpu().numpy() + 0.005) * 22000, a_min=0, a_max=255).astype("uint8").transpose((1, 2, 0))
temp = (normalize(result).permute((1, 2, 0)).detach().cpu().numpy() * 255).astype("uint8")
show_image(temp)
