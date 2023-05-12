# References
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    # https://github.com/jacobgil/vit-explain
    # https://jacobgil.github.io/deeplearning/vision-transformer-explainability#how-do-the-attention-activations-look-like-for-the-class-token-throughout-the-network- 
import torch
import torch.nn.functional as F
from typing import Literal
from PIL import Image
import numpy
import sys
import torchvision.transforms as T
import numpy as np
import cv2
import re
from itertools import product

from process_images import (
    load_image,
    show_image,
    save_image
)

IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHS = (IMG_SIZE // PATCH_SIZE) ** 2


class AttentionRollout:
    def __init__(
        self,
        model,
        head_fusion: Literal["mean", "max", "min", "sum"]="min",
        discard_ratio: float=0.9,
        attn_layer_regex=r"(.attn_drop)$"
    ):
        self.model = model.eval()
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.attn_mats = list()

        for name, module in self.model.named_modules():
            if re.search(pattern=attn_layer_regex, string=name):
                module.register_forward_hook(self.forward_hook_fn)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def forward_hook_fn(self, module, input, output):
        self.attn_mats.append(output.cpu())

    def _get_attention_matrices(self, image):
        with torch.no_grad():
            self.model(image)
        return self.attn_mats

    def get_attention_map(self, img):
        image = self.transform(_to_pil(img)).unsqueeze(0)
        attn_mats = attn_rollout._get_attention_matrices(image)

        attn_map = torch.eye(attn_mats[0].squeeze().shape[1])
        for attn_mat in attn_mats:
            # At every Transformer block we get an attention matrix $A_{ij}$
            # that defines how much attention is going to flow from token $j$ in the previous layer
            # to token $i$ in the next layer.

            # The Attention rollout paper suggests taking the average of the heads.
            # It can also make sense using other choices: like the minimum, the maximum, or using different weights. 
            if self.head_fusion == "mean":
                attn_mat = attn_mat.mean(dim=1)
            elif self.head_fusion == "min":
                attn_mat = attn_mat.min(dim=1)[0]
            elif self.head_fusion == "max":
                attn_mat = attn_mat.max(dim=1)[0]
            # 제시된 방법은 아니지만 `sum()`도 사용할 수 있을 것입니다.
            elif self.head_fusion == "sum":
                attn_mat = attn_mat.sum(dim=1)

            # Without discarding low attention pixels the attention map is very noisy
            # and doesn’t seem to focus only on the interesting part of the image.
            # The more pixels we remove, we are able to better isolate the salient object in the image.
            flattened = torch.flatten(attn_mat.squeeze(), start_dim=0, end_dim=-1)
            sorted, _ = torch.sort(flattened, dim=0)
            ref_val = sorted[int(len(sorted) * self.discard_ratio)]
            attn_mat.masked_fill_(mask=(attn_mat < ref_val), value=0)

            # We also have the residual connections.
            # We can model them by adding the identity matrix $I$ to the layer attention matrix: $A_{ij} + I$.
            id_mat = torch.eye(attn_mat.shape[1])
            attn_mat = attn_mat + id_mat

            # If we look at the first row (shape 197), and discard the first value (left with shape 196=14x14) that’s how the inforattn_mapion flows from the different locations in the image to the class token.
            # We also have to normalize the rows, to keep the total attention flow $1$.
            # 따라서 각 행마다 합을 구해야 함 
            attn_mat /= attn_mat.sum(dim=2)

            # Recursively multiply the attention matrix
            # attn_map = torch.matmul(attn_map, attn_mat)
            attn_map = torch.matmul(attn_mat, attn_map)

        attn_map = attn_map.squeeze()[0, 1:]
        # attn_map = attn_map.view(int(attn_map.shape[0] ** 0.5), int(attn_map.shape[0] ** 0.5))
        attn_map = attn_map.reshape(int(attn_map.shape[0] ** 0.5), int(attn_map.shape[0] ** 0.5))

        attn_map = attn_map.detach().cpu().numpy()
        attn_map -= attn_map.min()
        attn_map /= attn_map.max()
        attn_map *= 255
        attn_map = attn_map.astype("uint8")

        h, w = img.shape[: 2]
        resized = cv2.resize(src=attn_map, dsize=(w, h))
        return resized


if __name__ == "__main__":
    model = torch.hub.load("facebookresearch/deit:main", model="deit_tiny_patch16_224", pretrained=True)
    img = load_image("/Users/jongbeomkim/Desktop/workspace/generative_models/style_transfer/samples/golden_retriever.jpg")

    for head_fusion, discard_ratio in product(
        ["mean", "max", "min", "sum"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ):
        attn_rollout = AttentionRollout(model=model, head_fusion=head_fusion, discard_ratio=discard_ratio)
        attn_map = attn_rollout.get_attention_map(img=img)
        # show_image(attn_map, img)

        save_image(
            img1=img,
            img2=attn_map,
            alpha=0.7,
            path=f"""/Users/jongbeomkim/Desktop/workspace/explainable_ai/attention_rollout/attention_map_samples/head_fusion_{head_fusion}_discard_ratio_{discard_ratio}.jpg"""
        )




# def rollout(attentions, discard_ratio, head_fusion):
#     result = torch.eye(attentions[0].size(-1))
#     with torch.no_grad():
#         for attention in attentions:
#             # print(attention.shape)
#             if head_fusion == "mean":
#                 attention_heads_fused = attention.mean(axis=1)
#             elif head_fusion == "max":
#                 attention_heads_fused = attention.max(axis=1)[0]
#             elif head_fusion == "min":
#                 attention_heads_fused = attention.min(axis=1)[0]
#             else:
#                 raise "Attention head fusion type Not supported"

#             # Drop the lowest attentions, but
#             # don"t drop the class token
#             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#             _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
#             indices = indices[indices != 0]
#             flat[0, indices] = 0

#             # To account for residual connections, we add an identity matrix
#             # to the attention matrix and re-normalize the weights.
#             I = torch.eye(attention_heads_fused.size(-1))
#             a = (attention_heads_fused + 1.0*I)/2
#             a = a / a.sum(dim=-1)

#             result = torch.matmul(a, result)
    
#     # Look at the total attention between the class token,
#     # and the image patches
#     attn_map = result[0, 0 , 1 :]
#     # In case of 224x224 image, this brings us from 196 to 14
#     width = int(attn_map.size(-1)**0.5)
#     attn_map = attn_map.reshape(width, width).numpy()
#     attn_map = attn_map / np.max(attn_map)
#     return attn_map    

# class VITAttentionRollout:
#     def __init__(self, model, attn_layer_regex="attn_drop", head_fusion="mean",
#         discard_ratio=0.9):
#         self.model = model
#         self.head_fusion = head_fusion
#         self.discard_ratio = discard_ratio
#         for name, module in self.model.named_modules():
#             if attn_layer_regex in name:
#                 module.register_forward_hook(self.get_attention)

#         self.attn_mats = list()

#     def get_attention(self, module, input, output):
#         # print(output.shape, input[0].shape)
#         print(input[0].shape, output[0].shape)
#         self.attn_mats.append(output.cpu())

#     def __call__(self, input_tensor):
#         self.attn_mats = list()
#         with torch.no_grad():
#             self.model(input_tensor)
#         return rollout(self.attn_mats, self.discard_ratio, self.head_fusion)
# model = torch.hub.load("facebookresearch/deit:main", model="deit_tiny_patch16_224", pretrained=True)
# ar = VITAttentionRollout(model=model, attn_layer_regex="qkv")
# # ar = VITAttentionRollout(model=model, attn_layer_regex="attn_drop")
# out = ar(image)
# # out.shape
