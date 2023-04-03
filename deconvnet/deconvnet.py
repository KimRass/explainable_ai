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


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


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
        self.layer2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.layer4 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer5 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.layer7 = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.layer9 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer10 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.layer12 = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.layer14 = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.layer16 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer17 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.layer19 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer21 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer23 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layer24 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer26 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.layer28 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
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
    
    def _transfer_convolution_layer_parameters(self, convnet):
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


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = image.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    # copied_img *= variance
    # copied_img += mean
    # copied_img = copied_img.sum(axis=2)
    copied_img -= copied_img.min()
    copied_img /= copied_img.max()
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def tensor_to_array2(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = image.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_img *= variance
    copied_img += mean
    # copied_img = copied_img.sum(axis=2)
    # copied_img -= copied_img.min()
    # copied_img /= copied_img.max()
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _convert_to_pil(copied_img)
    copied_img.show()


if __name__ == "__main__":
    vgg16_pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # input = torch.randn((1, 3, 224, 224))
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((512, 512)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    # img = load_image("/Users/jongbeomkim/Downloads/imagenet-mini/val/n13133613/ILSVRC2012_val_00019877.JPEG")
    img = load_image("D:/imagenet-mini/train/n02088238/n02088238_1635.JPEG")
    image = transform(img).unsqueeze(0)

    pretrained_layers = vgg16_pretrained.features
    convnet = VGG16ConvNet()
    convnet._transfer_convolution_layer_parameters(pretrained_layers)
    output, maxpool_indices = convnet._get_output_and_maxpool_indices(image)
    # output.min(), output.max()
    
    deconvnet = VGG16DeconvNet()
    deconvnet._transfer_convolution_layer_parameters(convnet)
    temp = deconvnet(output, maxpool_indices)
    
    copied_img = temp.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_img.min(), copied_img.max()
    # copied_img *= variance
    # copied_img += mean
    # copied_img = copied_img.sum(axis=2)
    copied_img -= copied_img.min()
    copied_img /= copied_img.max()
    copied_img.min(), copied_img.max()
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    copied_img.shape
    show_image(copied_img)
    # temp = tensor_to_array(temp)
    # show_image(temp)
    
    # res = np.concatenate([tensor_to_array2(image), temp[..., None]], axis=2)
    # show_image(res)
    # img.shape, temp.shape

    
    show_image(tensor_to_array2(image))
