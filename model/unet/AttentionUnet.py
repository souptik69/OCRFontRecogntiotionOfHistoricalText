import torch
from torch import nn

from model.unet.UnetAttentionBlock import UnetAttentionBlock
from model.unet.UnetConvBlock import UnetConvBlock
from model.unet.UnetUpConvBlock import UnetUpConvBlock


class AttentionUnet(nn.Module):

    def __init__(self, img_ch=1, output_ch=1):
        super(AttentionUnet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = UnetConvBlock(img_ch, filters[0])
        self.Conv2 = UnetConvBlock(filters[0], filters[1])
        self.Conv3 = UnetConvBlock(filters[1], filters[2])
        self.Conv4 = UnetConvBlock(filters[2], filters[3])
        self.Conv5 = UnetConvBlock(filters[3], filters[4])

        self.Up5 = UnetUpConvBlock(filters[4], filters[3])
        self.Att5 = UnetAttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = UnetConvBlock(filters[4], filters[3])

        self.Up4 = UnetUpConvBlock(filters[3], filters[2])
        self.Att4 = UnetAttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = UnetConvBlock(filters[3], filters[2])

        self.Up3 = UnetUpConvBlock(filters[2], filters[1])
        self.Att3 = UnetAttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = UnetConvBlock(filters[2], filters[1])

        self.Up2 = UnetUpConvBlock(filters[1], filters[0])
        self.Att2 = UnetAttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = UnetConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)
        return out
