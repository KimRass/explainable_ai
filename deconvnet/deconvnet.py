# References
    # https://github.com/csgwon/pytorch-deconvnet/blob/master/models/vgg16_deconv.py
    # https://github.com/huybery/VisualizingCNN

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from process_images import (
    load_image,
    _to_pil,
    show_image
)


class DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.forward_net = self.VGG16ForwardNetwork()
        self.backward_net = self.VGG16BackwardNetwork()
    
    def transfer_parameters_to_forward_network(self, pretrained_layers):
        self.forward_net._transfer_conv2d_to_conv2d(pretrained_layers)
    
    def transfer_parameters_to_backward_network(self):
        self.backward_net._transfer_conv2d_to_convtranspose2d(self.forward_net)

    def _sort_feature_map(self, feat_map):
        argsort = torch.argsort(feat_map.sum(dim=(2, 3)), dim=1, descending=True)[0]
        feat_map = feat_map[:, argsort]
        return feat_map
    
    def project_feature_map(self, image, trg_layer, id=None):
        self.forward_net.register_hooks(trg_layer=trg_layer)
        maxpool_indices = self.forward_net._get_maxpool_indices(image)

        feat_map = self.forward_net.feat_map
        feat_map = self._sort_feature_map(feat_map)
        # return feat_map
        if id is not None:
            feat_map[:, : id, :, :] = 0
            feat_map[:, id + 1:, :, :] = 0

        x = self.backward_net(x=feat_map, maxpool_indices=maxpool_indices, trg_layer=trg_layer)
        return x

    class VGG16ForwardNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.pool_indices = dict()
            self.feat_maps = dict()
            self.feat_map = None

            self.layer0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.layer1 = nn.ReLU()
            self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.layer3 = nn.ReLU()
            self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.layer5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.layer6 = nn.ReLU()
            self.layer7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.layer8 = nn.ReLU()
            self.layer9 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.layer10 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.layer11 = nn.ReLU()
            self.layer12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.layer13 = nn.ReLU()
            self.layer14 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.layer15 = nn.ReLU()
            self.layer16 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.layer17 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.layer18 = nn.ReLU()
            self.layer19 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.layer20 = nn.ReLU()
            self.layer21 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.layer22 = nn.ReLU()
            self.layer23 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.layer24 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.layer25 = nn.ReLU()
            self.layer26 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.layer27 = nn.ReLU()
            self.layer28 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.layer29 = nn.ReLU()
            self.layer30 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        def forward(self, x):
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x, pool_indices = self.layer4(x)
            self.pool_indices[4] = pool_indices
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x, pool_indices = self.layer9(x)
            self.pool_indices[9] = pool_indices
            x = self.layer10(x)
            x = self.layer11(x)
            x = self.layer12(x)
            x = self.layer13(x)
            x = self.layer14(x)
            x = self.layer15(x)
            x, pool_indices = self.layer16(x)
            self.pool_indices[16] = pool_indices
            x = self.layer17(x)
            x = self.layer18(x)
            x = self.layer19(x)
            x = self.layer20(x)
            x = self.layer21(x)
            x = self.layer22(x)
            x, pool_indices = self.layer23(x)
            self.pool_indices[23] = pool_indices
            x = self.layer24(x)
            x = self.layer25(x)
            x = self.layer26(x)
            x = self.layer27(x)
            x = self.layer28(x)
            x = self.layer29(x)
            x, pool_indices = self.layer30(x)
            self.pool_indices[30] = pool_indices
            return x
            
        def _transfer_conv2d_to_conv2d(self, pretrained_layers):
            for pretrained_layer, learning_layer in zip(pretrained_layers, self.children()):
                if isinstance(pretrained_layer, nn.Conv2d) and isinstance(learning_layer, nn.Conv2d):
                    learning_layer.weight.data = pretrained_layer.weight.data
                    learning_layer.bias.data = pretrained_layer.bias.data

        
        def _get_maxpool_indices(self, x):
            x = self(x)
            return self.pool_indices
        
        def register_hooks(self, trg_layer):
            def get_feature_map(name, trg_layer):
                def forward_hook_fn(module, input, output):
                    if name == trg_layer:
                        self.feat_map = output
                    else:
                        return None
                return forward_hook_fn

            for name, module in self.named_modules():
                if not isinstance(module, nn.Sequential):
                    module.register_forward_hook(
                        get_feature_map(name=name, trg_layer=trg_layer)
                    )

    class VGG16BackwardNetwork(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer0 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
            self.layer1 = nn.ReLU()
            self.layer2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
            self.layer3 = nn.ReLU()
            self.layer4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.layer5 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
            self.layer6 = nn.ReLU()
            self.layer7 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
            self.layer8 = nn.ReLU()
            self.layer9 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.layer10 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
            self.layer11 = nn.ReLU()
            self.layer12 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
            self.layer13 = nn.ReLU()
            self.layer14 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
            self.layer15 = nn.ReLU()
            self.layer16 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.layer17 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
            self.layer18 = nn.ReLU()
            self.layer19 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
            self.layer20 = nn.ReLU()
            self.layer21 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
            self.layer22 = nn.ReLU()
            self.layer23 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.layer24 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
            self.layer25 = nn.ReLU()
            self.layer26 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
            self.layer27 = nn.ReLU()
            self.layer28 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
            self.layer29 = nn.ReLU()
            self.layer30 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        def forward(self, x, maxpool_indices, trg_layer):
            layer_num = int(trg_layer[5:])
            for i in range(layer_num, -1, -1):
                print(f"""self.layer{i}""")
                layer = eval(f"""self.layer{i}""")
                if isinstance(layer, nn.MaxUnpool2d):
                    x = layer(x, maxpool_indices[i])
                else:
                    x = layer(x)
            # x = self.layer30(x, maxpool_indices[30])
            # x = self.layer29(x)
            # x = self.layer28(x)
            # x = self.layer27(x)
            # x = self.layer26(x)
            # x = self.layer25(x)
            # x = self.layer24(x)
            # x = self.layer23(x, maxpool_indices[23])
            # x = self.layer22(x)
            # x = self.layer21(x)
            # x = self.layer20(x)
            # x = self.layer19(x)
            # x = self.layer18(x)
            # x = self.layer17(x)
            # x = self.layer16(x, maxpool_indices[16])
            # x = self.layer15(x)
            # x = self.layer14(x)
            # x = self.layer13(x)
            # x = self.layer12(x)
            # x = self.layer11(x)
            # x = self.layer10(x)
            # x = self.layer9(x, maxpool_indices[9])
            # x = self.layer8(x)
            # x = self.layer7(x)
            # x = self.layer6(x)
            # x = self.layer5(x)
            # x = self.layer4(x, maxpool_indices[4])
            # x = self.layer3(x)
            # x = self.layer2(x)
            # x = self.layer1(x)
            # x = self.layer0(x)
            return x
        
        def _transfer_conv2d_to_convtranspose2d(self, convnet):
            for deconvnet_layer, convnet_layer in (
                (self.layer0, convnet.layer0),
                (self.layer2, convnet.layer2),
                (self.layer5, convnet.layer5),
                (self.layer7, convnet.layer7),
                (self.layer10, convnet.layer10),
                (self.layer12, convnet.layer12),
                (self.layer14, convnet.layer14),
                (self.layer17, convnet.layer17),
                (self.layer19, convnet.layer19),
                (self.layer21, convnet.layer21),
                (self.layer24, convnet.layer24),
                (self.layer26, convnet.layer26),
                (self.layer28, convnet.layer28),
            ):
                deconvnet_layer.weight.data = convnet_layer.weight.data
            for deconvnet_layer, convnet_layer in (
                (self.layer2, convnet.layer0),
                (self.layer5, convnet.layer2),
                (self.layer7, convnet.layer5),
                (self.layer10, convnet.layer7),
                (self.layer12, convnet.layer10),
                (self.layer14, convnet.layer12),
                (self.layer17, convnet.layer14),
                (self.layer19, convnet.layer17),
                (self.layer21, convnet.layer19),
                (self.layer24, convnet.layer21),
                (self.layer26, convnet.layer24),
                (self.layer28, convnet.layer26),
            ):
                deconvnet_layer.bias.data = convnet_layer.bias.data


def print_all_layers(model):
    print(f"""|         NAME         |                            MODULE                            |""")
    print(f"""|----------------------|--------------------------------------------------------------|""")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and name:
            print(f"""| {name:20s} | {str(type(module)):60s} |""")


def postprocess_reconstructed_activation(act, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_act = act.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    # copied_act *= 100
    copied_act -= copied_act.min()
    copied_act /= copied_act.max()
    copied_act *= 255.0
    copied_act = np.clip(a=copied_act, a_min=0, a_max=255).astype("uint8")
    return copied_act


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            # T.Resize((512, 512)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    img = load_image("/Users/jongbeomkim/Downloads/imagenet-mini/train/n07697313/n07697313_363.JPEG")
    # img = load_image("D:/imagenet-mini/train/n02088238/n02088238_1635.JPEG")
    image = transform(img).unsqueeze(0)

    deconvnet = DeconvNet()
    vgg16_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    deconvnet.transfer_parameters_to_forward_network(vgg16_pretrained.features)
    deconvnet.transfer_parameters_to_backward_network()

    act = deconvnet.project_feature_map(image=image, trg_layer="layer7", id=0)
    result = postprocess_reconstructed_activation(act)
    show_image(result)
