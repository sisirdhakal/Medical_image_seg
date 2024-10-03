import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim

import torchvision.models.segmentation as segmentation
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image


from torchvision.io.image import read_image
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)
from torchvision.transforms.functional import to_pil_image


# defining a double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # creating a sequential block of two convolutions with batch normalization and ReLU activation
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # applying the convolution block to the input
        return self.Conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.downs = nn.ModuleList()  # list for downsampling layers
        self.ups = nn.ModuleList()  # list for upsampling layers
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # pooling operation for downsampling

        # creating downsampling layers
        for feature in features:
            self.downs.append(
                DoubleConv(in_channels, feature)
            )  # adding double conv layer
            in_channels = feature  # updating input channels

        # creating upsampling layers
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )  # adding transpose convolution for upsampling
            self.ups.append(
                DoubleConv(feature * 2, feature)
            )  # adding double conv after upsampling

        self.bottomlayer = DoubleConv(
            features[-1], features[-1] * 2
        )  # bottleneck layer
        self.finallayer = nn.Conv2d(
            features[0], out_channels, kernel_size=1
        )  # final output layer

    def forward(self, x):
        skip_connections = []  # list to hold skip connections

        # going through the downward path
        for i, down in enumerate(self.downs):
            x = down(x)  # processing through the down layer
            skip_connections.append(x)  # storing the output for skip connection
            x = self.pool(x)  # pooling to reduce spatial dimensions

        x = self.bottomlayer(x)  # bottleneck layer output

        skip_connections = skip_connections[::-1]  # reversing for upward path

        outputs = []  # list to hold outputs for deep supervision

        # going through the upward path
        for j in range(len(self.ups) // 2):
            x = self.ups[j * 2](x)  # applying transpose convolution (upsampling)
            skip_connection = skip_connections[
                j
            ]  # retrieving the corresponding skip connection

            # resizing the output to match the skip connection dimensions if needed
            if x.shape != skip_connection.shape:
                x = torchvision.transforms.functional.resize(
                    x, size=skip_connection.shape[2:]
                )

            concat_skip = torch.cat(
                (skip_connection, x), dim=1
            )  # concatenating the skip connection
            x = self.ups[j * 2 + 1](concat_skip)  # processing concatenated output

            # adding output for deep supervision
            outputs.append(self.finallayer(x))  # collecting outputs for supervision

        return outputs  # returning outputs from all nested levels


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor  # scaling factor for upsampling

    def forward(self, x):
        # applying bilinear interpolation for upsampling
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
        )


# Nested Unet model


class NestedUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, deep_supervision=False, mode="accurate"):
        super(NestedUNet, self).__init__()

        self.deep_supervision = deep_supervision
        n1 = 64  # initial number of filters
        filters = [
            n1,
            n1 * 2,
            n1 * 4,
            n1 * 8,
            n1 * 16,
        ]  # defining filter sizes for each layer
        self.mode = mode  # mode for output selection

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # pooling operation for downsampling
        self.Up = Upsample(scale_factor=2)  # upsampling operation

        # defining encoder layers
        self.conv0_0 = DoubleConv(in_ch, filters[0])  # first layer
        self.conv1_0 = DoubleConv(filters[0], filters[1])  # second layer
        self.conv2_0 = DoubleConv(filters[1], filters[2])  # third layer
        self.conv3_0 = DoubleConv(filters[2], filters[3])  # fourth layer
        self.conv4_0 = DoubleConv(filters[3], filters[4])  # fifth layer

        # defining nested layers

        # first level of nesting
        self.conv0_1 = DoubleConv(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv(filters[3] + filters[4], filters[3])

        # second level of nesting
        self.conv0_2 = DoubleConv(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = DoubleConv(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = DoubleConv(filters[2] * 2 + filters[3], filters[2])

        # third level of nesting
        self.conv0_3 = DoubleConv(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = DoubleConv(filters[1] * 3 + filters[2], filters[1])

        # fourth level of nesting
        self.conv0_4 = DoubleConv(filters[0] * 4 + filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)  # final output layer

        # list to hold segmentation outputs for deep supervision
        self.seg_outputs = []

    def forward(self, x):
        # Encoder path
        x0_0 = self.conv0_0(x)  # processing input through the first encoder layer
        x1_0 = self.conv1_0(
            self.pool(x0_0)
        )  # processing through second layer after pooling
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))  # first nested output

        x2_0 = self.conv2_0(
            self.pool(x1_0)
        )  # processing through third layer after pooling
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))  # second nested output
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.Up(x1_1)], 1)
        )  # second nested output

        x3_0 = self.conv3_0(
            self.pool(x2_0)
        )  # processing through fourth layer after pooling
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))  # third nested output
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.Up(x2_1)], 1)
        )  # third nested output
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1)
        )  # third nested output

        x4_0 = self.conv4_0(
            self.pool(x3_0)
        )  # processing through fifth layer after pooling
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))  # fourth nested output
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self.Up(x3_1)], 1)
        )  # fourth nested output
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1)
        )  # fourth nested output
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1)
        )  # fourth nested output

        # collecting outputs for deep supervision
        if self.deep_supervision:
            self.seg_outputs.append(x0_1)  # storing first nested output
            self.seg_outputs.append(x0_2)  # storing second nested output
            self.seg_outputs.append(x0_3)  # storing third nested output
            self.seg_outputs.append(x0_4)  # storing fourth nested output

        # final segmentation output
        output = self.final(x0_4)  # applying final convolution to the last layer output

        # handling deep supervision outputs
        if self.deep_supervision:
            if self.mode == "accurate":
                # averaging all outputs for accurate mode
                final_output = torch.mean(torch.stack(self.seg_outputs), dim=0)
                return final_output  # returning averaged output
            elif self.mode == "fast":
                # selecting one output for fast mode
                final_output = self.seg_outputs[-1]  # here, choosing the last output
                return final_output  # returning selected output

        return output  # returning final output if no deep supervision is used
