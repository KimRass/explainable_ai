# References
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    # https://github.com/jacobgil/vit-explain

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

IMG_SIZE = 224
PATCH_SIZE = 16
N_PATCHS = (IMG_SIZE // PATCH_SIZE) ** 2


class AttentionRollout:
    def __init__(
        self,
        model,
        head_fusion: str = Literal["mean", "max", "min"],
        discard_ratio=0.9,
        attn_layer_regex=r"(.attn_drop)$"
    ):
        self.model = model.eval()
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

        self.attn_maps = list()

        for name, module in self.model.named_modules():
            if re.search(pattern=attn_layer_regex, string=name):
                module.register_forward_hook(self.forward_hook_fn)

    def forward_hook_fn(self, module, input, output):
        self.attn_maps.append(output.cpu())

    def _get_attention_maps(self, image):
        with torch.no_grad():
            self.model(image)
        return self.attn_maps

model = torch.hub.load("facebookresearch/deit:main", model="deit_tiny_patch16_224", pretrained=True)
attn_rollout = AttentionRollout(model=model)

img = load_image("/Users/jongbeomkim/Desktop/workspace/generative_models/style_transfer/samples/golden_retriever.jpg")
# image = Image.open("/Users/jongbeomkim/Desktop/workspace/generative_models/style_transfer/samples/golden_retriever.jpg")
image = _to_pil(img)
transform = T.Compose([
    T.ToTensor(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image = transform(image).unsqueeze(0)

attn_maps = attn_rollout._get_attention_maps(image)
mat = torch.ones_like(attn_maps[0][0, 0, ...])
for attn_map in attn_maps:
    # flat = torch.flatten(attn_map, start_dim=1, end_dim=-1)
    # _, ids = torch.topk(flat, k=int(flat.size(-1) * 0.9), dim=1, sorted=False)
    # ids
    # ids[ids != 0]

    attn_map = attn_map.squeeze(0)
    attn_map = attn_map.sum(dim=0)

    # To account for residual connections, we add an identity matrix
    # to the attention matrix and re-normalize the weights.
    residual_attn = torch.eye(attn_map.shape[0])
    attn_map = attn_map + residual_attn

    attn_map /= attn_map.sum(dim=1)

    # Recursively multiply the weight matrices
    mat = torch.matmul(mat, attn_map)

mask = mat[0, 1:]
mask = mask.view(int(mask.shape[0] ** 0.5), int(mask.shape[0] ** 0.5))

arr = mask.detach().cpu().numpy()
arr *= arr.max()
arr = arr.astype("uint8")

resized = cv2.resize(src=arr, dsize=(IMG_SIZE, IMG_SIZE))
show_image(cv2.resize(src=img, dsize=(IMG_SIZE, IMG_SIZE)), resized)
    



def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            # print(attention.shape)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don"t drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            # To account for residual connections, we add an identity matrix
            # to the attention matrix and re-normalize the weights.
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attn_layer_regex="attn_drop", head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attn_layer_regex in name:
                module.register_forward_hook(self.get_attention)

        self.attn_maps = list()

    def get_attention(self, module, input, output):
        # print(output.shape, input[0].shape)
        print(input[0].shape, output[0].shape)
        self.attn_maps.append(output.cpu())

    def __call__(self, input_tensor):
        self.attn_maps = list()
        with torch.no_grad():
            self.model(input_tensor)
        return rollout(self.attn_maps, self.discard_ratio, self.head_fusion)
model = torch.hub.load("facebookresearch/deit:main", model="deit_tiny_patch16_224", pretrained=True)
ar = VITAttentionRollout(model=model, attn_layer_regex="qkv")
# ar = VITAttentionRollout(model=model, attn_layer_regex="attn_drop")
out = ar(image)
# out.shape
