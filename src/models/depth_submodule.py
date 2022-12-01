from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left = F.pad(
            torch.index_select(left, 3, Variable(torch.LongTensor([i for i in range(shift, width)])).cuda()),
            (shift, 0, 0, 0))
        shifted_right = F.pad(
            torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width - shift)])).cuda()),
            (shift, 0, 0, 0))
        out = torch.cat((shifted_left, shifted_right), 1).view(batch, filters * 2, 1, height, width)
        return out

class depthregression(nn.Module):
    def __init__(self, maxdepth):
        super(depthregression, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(1, 1+maxdepth)), [1, maxdepth, 1, 1])).cuda(),
        #                      requires_grad=False)
        self.disp = torch.arange(1, 1+maxdepth, device='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * self.disp, 1)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        # self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(),
        #                      requires_grad=False)
        self.disp = torch.arange(maxdisp, devices='cuda', requires_grad=False).float()[None, :, None, None]

    def forward(self, x):
        # disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * self.disp, 1)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x): # [4, 3, 256, 512]
        output = self.firstconv(x) # [4, 32, 128, 256]

        output = self.layer1(output) # [4, 32, 128, 256]
        output_raw = self.layer2(output) # [4, 64, 64, 128]
        output = self.layer3(output_raw) # [4, 128, 64, 128]
        output_skip = self.layer4(output) # [4, 128, 64, 128]

        output_branch1 = self.branch1(output_skip) # [4, 32, 1, 2]
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear') # [4, 32, 64, 128]

        output_branch2 = self.branch2(output_skip) # [4, 32, 2, 4]
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear') # [4, 32, 64, 128]

        output_branch3 = self.branch3(output_skip) # [4, 32, 4, 8]
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear') # [4, 32, 64, 128]

        output_branch4 = self.branch4(output_skip) # [4, 32, 8, 16]
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear') # [4, 32, 64, 128]

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1) # [4, 320, 64, 128]
        output_feature = self.lastconv(output_feature) # [4, 32, 64, 128]

        return output_feature

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1)

        self.ds = convbn(inplanes, planes, 3, stride)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.ds(x)
        out += x
        out = self.relu(out)
        return out

class UpProject(nn.Module):

    # def __init__(self, in_channels, out_channels, batch_size):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()
        # self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))  # author's interleaving pading in github

        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))  # author's interleaving pading in github

        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))  # author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))  # author's interleaving pading in github

        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))  # author's interleaving pading in github

        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))  # author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        # out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
        #     self.batch_size, -1, height, width * 2)
        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            x.size()[0], -1, height, width * 2)
        # out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
        #     self.batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            x.size()[0], -1, height, width * 2)

        # out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
        #     self.batch_size, -1, height * 2, width * 2)
        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            x.size()[0], -1, height * 2, width * 2)

        # out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
        #     self.batch_size, -1, height, width * 2)
        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            x.size()[0], -1, height, width * 2)
        # out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
        #     self.batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            x.size()[0], -1, height, width * 2)

        # out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
        #     self.batch_size, -1, height * 2, width * 2)
        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            x.size()[0], -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out

class feature_extraction_fusion(nn.Module):
    def __init__(self):
        super(feature_extraction_fusion, self).__init__()
        self.batch_size = 4
        self.convS = ResBlock(2, 32, 1)
        self.convS0 = ResBlock(32, 97, 1)
        self.convS1 = ResBlock(97, 193, 2)
        self.convS2 = ResBlock(193, 385, 2)
        self.convS3 = ResBlock(385, 513, 2)
        self.convS4 = ResBlock(513, 512, 2)

        self.conv1 = ResBlock(3, 32, 1)
        self.conv2 = ResBlock(32, 64, 1)
        self.conv3 = ResBlock(64, 128, 2)
        self.conv3_1 = ResBlock(128, 128, 1)
        self.conv4 = ResBlock(128, 256, 2)
        self.conv4_1 = ResBlock(256, 256, 1)
        self.conv5 = ResBlock(256, 256, 2)
        self.conv5_1 = ResBlock(256, 256, 1)
        self.conv6 = ResBlock(256, 512, 2)
        self.conv6_1 = ResBlock(512, 512, 1)

        self.deconv5 = self._make_upproj_layer(UpProject, 512, 256)  # # TODO batch size
        self.deconv4 = self._make_upproj_layer(UpProject, 513, 128)  #

        self.predict_normal6 = predict_normal(512)  #
        self.predict_normal5 = predict_normal(513)  #

        self.upsampled_normal6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)  #
        self.upsampled_normal5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)  #

        self.reduce_feature = reduce_feature(385)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    # def _make_upproj_layer(self, block, in_channels, out_channels, bs=1):
    def _make_upproj_layer(self, block, in_channels, out_channels):

        # print("11: ", type(block))
        # print("block: ", block)
        # print("out_conv6 block type: {}, out_conv6 block shape: {}".format(type(block), block.shape))
        # return block(in_channels, out_channels, bs)
        return block(in_channels, out_channels)

    # def forward(self, x): # [4, 3, 256, 512]
    def forward(self, img, sparse, mask):  # [4, 3, 256, 512]
        """

        :param img: [bs, 3, H, W]
        :param sparse: [bs, 1, H, W]
        :param mask: [bs, 1, H, W]
        :return:
        """
        # mask + sparse
        inputM = mask
        inputS = torch.cat((sparse, inputM), 1)  # [bs, 2, 256, 512]
        inputS_conv = self.convS(inputS)  # [bs, 32, 256, 512]
        inputS_conv0 = self.convS0(inputS_conv)  # [bs, 97, 256, 512]
        # 2
        inputS_conv1 = self.convS1(inputS_conv0)  # [bs, 193, 128, 256]
        # 3
        inputS_conv2 = self.convS2(inputS_conv1)  # [bs, 385, 64, 128]
        # 4
        inputS_conv3 = self.convS3(inputS_conv2)  # [bs, 513, 32, 64]
        # 5
        inputS_conv4 = self.convS4(inputS_conv3)  # [bs, 512, 16, 32]

        # RGB
        # 0
        input2 = img  # [bs, 3, 256, 512]
        # 1
        out_conv2 = self.conv2(self.conv1(input2))  # [bs, 64, 256, 512]
        # 2
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  # [bs, 128, 128, 256]
        # 3
        out_conv4 = self.conv4_1(self.conv4(out_conv3))  # [bs, 256, 64, 128]
        # 4
        out_conv5 = self.conv5_1(self.conv5(out_conv4))  # [bs, 256, 32, 64]
        # 5
        out_conv6 = self.conv6_1(self.conv6(out_conv5))  # [bs, 512, 16, 32] + [bs, 512, 16, 32]

        out_conv6 = out_conv6 + inputS_conv4 # TODO: concat or sum

        out6 = self.predict_normal6(out_conv6)  # conv 2D [bs, 1, 16, 32]
        # up-projection 5
        normal6_up = self.upsampled_normal6_to_5(out6)  # ConvTranspose2d [bs, 1, 32, 64]
        out_deconv5 = self.deconv5(out_conv6)  # conv 2D [bs, 256, 32, 64]

        # ([bs, 256, 32, 64], [bs, 256, 32, 64], [bs, 1, 32, 64])--> [bs, 256+256+1, 16, 32] + [bs, 513, 32, 64]
        # concat features from the RGB/normal but sum the features from the sparse depth onto the features in decoder
        concat5 = adaptative_cat(out_conv5, out_deconv5, normal6_up) + inputS_conv3  # concat [bs, 513, 32, 64]
        out5 = self.predict_normal5(concat5)  # conv 2D [bs, 1, 32, 64]
        normal5_up = self.upsampled_normal5_to_4(out5)  # up-projection 4 [bs, 1, 64, 128]
        out_deconv4 = self.deconv4(concat5)  # [bs, 128, 64, 128]

        concat4 = adaptative_cat(out_conv4, out_deconv4, normal5_up) + inputS_conv2  # [bs, 385, 64, 128]

        out_decoder = self.reduce_feature(concat4)
        return out_decoder


def convbn(in_planes, out_planes, kernel_size, stride, pad=1, dilation=1): # TODO: i have added pad and dilatation default
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


def predict_normal(in_planes):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    :param in_planes:
    :return:
    """
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def adaptative_cat(out_conv, out_deconv, out_depth_up):
    """

    :param out_conv: [bs, 256, 16, 32]
    :param out_deconv: [bs, 256, 16, 32]
    :param out_depth_up: [bs, 3, 16, 32]
    :return:
    """
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)] # [bs, 256, 16, 32]
    # print(out_deconv.shape)
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)] # [bs, 3, 16, 32]
    # print(out_depth_up.shape)
    # print(torch.cat((out_conv, out_deconv, out_depth_up), 1).shape)
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)  # [bs, 515, 16, 32]

def reduce_feature(in_planes):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    :param in_planes:
    :return:
    """
    return nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1, bias=True)









