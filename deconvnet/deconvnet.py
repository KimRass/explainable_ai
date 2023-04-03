# References
    # https://github.com/csgwon/pytorch-deconvnet/blob/master/models/vgg16_deconv.py

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as T
import numpy as np


class VGG16ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pool_indices = dict()
        # self._transfer_convolution_layer_parameters(pretrained_layers)

        # features = torch.nn.Sequential(
        self.layer0 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer5 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer6 = torch.nn.ReLU()
        self.layer7 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.layer8 = torch.nn.ReLU()
        self.layer9 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer10 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.layer11 = torch.nn.ReLU()
        self.layer12 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer13 = torch.nn.ReLU()
        self.layer14 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer15 = torch.nn.ReLU()
        self.layer16 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer17 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.layer18 = torch.nn.ReLU()
        self.layer19 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer20 = torch.nn.ReLU()
        self.layer21 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer22 = torch.nn.ReLU()
        self.layer23 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.layer24 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer25 = torch.nn.ReLU()
        self.layer26 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer27 = torch.nn.ReLU()
        self.layer28 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.layer29 = torch.nn.ReLU()
        self.layer30 = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
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
        
    def _transfer_convolution_layer_parameters(self, pretrained_layers):
        for pretrained_layer, learning_layer in zip(pretrained_layers, self.children()):
            if isinstance(pretrained_layer, nn.Conv2d) and isinstance(learning_layer, nn.Conv2d):
                print(pretrained_layer, learning_layer)
                learning_layer.weight.data = pretrained_layer.weight.data
                learning_layer.bias.data = pretrained_layer.bias.data
    
    def _get_output_and_maxpool_indices(self, x):
        x = self(x)
        return x, self.pool_indices


class VGG16DeconvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # self._transfer_convolution_layer_parameters(pretrained_layers)

        self.layer0 = torch.nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.layer1 = torch.nn.ReLU()
        self.layer2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.layer3 = torch.nn.ReLU()
        self.layer4 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer5 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.layer6 = torch.nn.ReLU()
        self.layer7 = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.layer8 = torch.nn.ReLU()
        self.layer9 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer10 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.layer11 = torch.nn.ReLU()
        self.layer12 = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.layer13 = torch.nn.ReLU()
        self.layer14 = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.layer15 = torch.nn.ReLU()
        self.layer16 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer17 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.layer18 = torch.nn.ReLU()
        self.layer19 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer20 = torch.nn.ReLU()
        self.layer21 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer22 = torch.nn.ReLU()
        self.layer23 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer24 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer25 = torch.nn.ReLU()
        self.layer26 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer27 = torch.nn.ReLU()
        self.layer28 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer29 = torch.nn.ReLU()
        self.layer30 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, maxpool_indices):
        x = self.layer30(x, maxpool_indices[30])
        x = self.layer28(x)
        x = self.layer26(x)
        x = self.layer24(x)
        x = self.layer23(x, maxpool_indices[23])
        x = self.layer21(x)
        x = self.layer19(x)
        x = self.layer17(x)
        x = self.layer16(x, maxpool_indices[16])
        x = self.layer14(x)
        x = self.layer12(x)
        x = self.layer10(x)
        x = self.layer9(x, maxpool_indices[9])
        x = self.layer7(x)
        x = self.layer5(x)
        x = self.layer4(x, maxpool_indices[4])
        x = self.layer2(x)
        x = self.layer0(x)
        return x
    
    def _transfer_convolution_layer_parameters(self, pretrained_layers):
        for pretrained_layer, learning_layer in zip(pretrained_layers, self.children()):
            if isinstance(pretrained_layer, nn.Conv2d) and isinstance(learning_layer, nn.ConvTranspose2d):
                print(pretrained_layer, learning_layer)
                learning_layer.weight.data = pretrained_layer.weight.data
                learning_layer.bias.data = pretrained_layer.bias.data


def denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= variance
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


if __name__ == "__main__":
    vgg16_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # input = torch.randn((1, 3, 224, 224))
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    img = load_image("/Users/jongbeomkim/Downloads/imagenet-mini/val/n13133613/ILSVRC2012_val_00019877.JPEG")
    image = transform(img).unsqueeze(0)

    pretrained_layers = vgg16_pretrained.features
    pretrained_layers[0]
    convnet = VGG16ConvNet()
    convnet._transfer_convolution_layer_parameters(pretrained_layers)
    output, maxpool_indices = convnet._get_output_and_maxpool_indices(image)

    deconvnet = VGG16DeconvNet()
    deconvnet._transfer_convolution_layer_parameters(pretrained_layers)
    
    temp = deconvnet(output, maxpool_indices)
    temp = temp.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    temp = denormalize_array(temp)
    show_image(temp)
    # layer30 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
    # layer28 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
    # temp = layer30(output, maxpool_indices[30])
    # layer28(temp)

    # for pretrained_layer, learning_layer in zip(vgg16_pretrained.features, deconvnet.children()):
    #     if isinstance(pretrained_layer, nn.Conv2d) and isinstance(learning_layer, nn.ConvTranspose2d):
    #         learning_layer.weight.data = pretrained_layer.weight.data
    #         learning_layer.bias.data = pretrained_layer.bias.data


    # input = torch.torch.tensor([[[[ 1.,  2.,  3., 4., 5.],
    #                             [ 6.,  7.,  8., 9., 10.],
    #                             [11., 12., 13., 14., 15.],
    #                             [16., 17., 18., 19., 20.]]]])
    # nn.MaxPool2d(kernel_size=2, stride=2)(input)