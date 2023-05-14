# References
    # https://stackoverflow.com/questions/55594969/how-to-visualise-filters-in-a-cnn-with-pytorch
    # https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/cnn_layer_visualization.py
    # https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
    # https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights, vgg19_bn, VGG19_BN_Weights
import matplotlib.pyplot as plt

from process_images import (
    denormalize_array,
    _to_3d
)


def convert_tensor_to_array(tensor):
    if tensor.ndim == 3:
        copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    elif tensor.ndim == 2:
        copied_tensor = tensor.clone().squeeze().detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def visualize_kernels(kernels):
    grid = torchvision.utils.make_grid(tensor=kernels, nrow=int(kernels.shape[0] ** 0.5), normalize=True, padding=3)
    grid = torch.sum(grid, axis=0)
    grid = convert_tensor_to_array(grid)
    grid = _to_3d(grid)
    return grid


class FeatureMapExtractor():
    def __init__(self, model):
        self.model = model

        self.feat_maps = list()

    def get_feature_maps(self, image):
        def forward_hook_fn(module, input, output):
            self.feat_maps.append(output)

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                print(module)
                module.register_forward_hook(forward_hook_fn)

        self.model(image)
        return self.feat_maps


if __name__ == "__main__":
    conv_layer = cnn.features[0]
    kernels = conv_layer.weight.data
    grid = visualize_kernels(kernels)
    show_image(grid)

    cnn = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    img = load_image("/Users/jongbeomkim/Downloads/download.png")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((256, 256)),
            # T.Normalize(mean=0., std=1.)
        ]
    )
    image = transform(img).unsqueeze(0)

    feat_map_extr = FeatureMapExtractor(cnn)
    feat_maps = feat_map_extr.get_feature_maps(image)

    feat_maps[0].shape
    grid = visualize_kernels(feat_maps[1])
    show_image(grid)