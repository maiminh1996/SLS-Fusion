import math
import torch.utils.data

from .depth_submodule import *

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class SLSFusion(nn.Module):
    def __init__(self, maxdisp, maxdepth, down=2):
        """
        init for fusion depth net: fusion RGB + lidar --> stereo
        :param maxdisp: 192 maxium disparity, the range of the disparity cost volume: [0, 192-1]
        :param maxdepth: the range of the depth cost volume: [1, 80]
        :param down: reduce x times resolution when build the depth cost volume
        """
        super(SLSFusion, self).__init__()
        self.maxdisp = maxdisp
        self.maxdepth = maxdepth
        self.down = down

        self.feature_extraction = feature_extraction_fusion()

        # self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
        self.dres0 = nn.Sequential(convbn_3d(128, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def warp(self, x, calib):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, D, H, W] (im2) [bs, 32+32, 192/4, 64, 128]
        flo: [B, 2, H, W] flow
        """
        # B,C,D,H,W to B,H,W,C,D
        x = x.transpose(1, 3).transpose(2, 4) # [bs, 64, 128, 32+32, 192/4]
        B, H, W, C, D = x.size()
        x = x.view(B, -1, C, D) # [bs, 8192, 32+32, 192/4]
        # mesh grid [bs, 40]
        xx = (calib / (self.down * 4.))[:, None] / torch.arange(1, 1 + self.maxdepth // self.down,
                                                                device='cuda').float()[None, :]
        new_D = self.maxdepth // self.down # 80//2
        xx = xx.view(B, 1, new_D).repeat(1, C, 1) # [bs, 64, 40]
        xx = xx.view(B, C, new_D, 1) # [bs, 64, 40, 1]

        yy = torch.arange(0, C, device='cuda').view(-1, 1).repeat(1, new_D).float() # [64, 40]
        yy = yy.view(1, C, new_D, 1).repeat(B, 1, 1, 1) # [bs, 64, 40, 1]

        grid = torch.cat((xx, yy), -1).float() # [bs, 64, 40, 2]

        vgrid = Variable(grid)

        # scale grid to [-1,1]
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(D - 1, 1) - 1.0 # [bs, 64, 40, 2]
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(C - 1, 1) - 1.0 # [bs, 64, 40, 2]

        if float(torch.__version__[:3])>1.2:
            output = nn.functional.grid_sample(x, vgrid, align_corners=True).contiguous() # [bs, 8192, 64, 40]
        else:
            output = nn.functional.grid_sample(x, vgrid).contiguous()
        output = output.view(B, H, W, C, new_D).transpose(1, 3).transpose(2, 4) # [bs, 64, 40, 64, 128]
        return output.contiguous() # [bs, 64, 40, 64, 128]

    def forward(self, left, right, sparse_left, sparse_right, mask_left, mask_right, calib):
        """

        :param left:
        :param right:
        :param calib:
        :return:
        """
        ##### feature extraction #####
        # calculate feature for left image [4, 3, 256, 512]
        # refimg_fea = self.feature_extraction(left) # [1,32,64,128]
        refimg_fea = self.feature_extraction(left, sparse_left, mask_left)  # [1,64,64,128]
        # print(refimg_fea.shape)
        targetimg_fea = self.feature_extraction(right, sparse_right, mask_right)  # [1,64,64,128]
        # print("targetimg_fea: {}".format(targetimg_fea.shape))
        cost = Variable(
            torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4,
                                   refimg_fea.size()[2],
                                   refimg_fea.size()[3]).zero_())
        # print("cost: {}".format(cost.shape))

        for i in range(self.maxdisp // 4): # /4 because H/4, W/4 --> disp/4
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:] # dich theo chieu ngang
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                # [:, refimage:targeting, 0, :, :]
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea # [bs, 64, 48, 64, 128] vs [bs, 32, 64, 128]
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        #####################################

        ##### Convert into depth cost volume #####
        # # [1, 80]: depth grid
        cost = cost.contiguous() # [bs, 64*2, 192/4, 64, 128]
        # print(cost.shape)

        cost = self.warp(cost, calib) # [bs, 64*2, 40, 64, 128]
        # print(cost.shape)

        ##########################################

        ##### conv 3D #####
        cost0 = self.dres0(cost) # conv 3D # [bs, 32, 40, 64, 128] # TODO: check: i have modified here 64 -->128
        cost0 = self.dres1(cost0) + cost0 # [bs, 32, 40, 64, 128]
        ###################

        ##### hourglass #####
        out1, pre1, post1 = self.dres2(cost0, None, None) # hourglass [bs, 32, 40, 64, 128], [bs, 64, 20, 32, 64], [bs, 64, 20, 32, 64]
        out1 = out1 + cost0 # [bs, 32, 40, 64, 128]

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0 # [bs, 32, 40, 64, 128]

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0 # [bs, 32, 40, 64, 128]
        #####################

        ##### conv 3D #####
        cost1 = self.classif1(out1) # conv 3D [bs, 1, 40, 64, 128]
        cost2 = self.classif2(out2) + cost1  # [bs, 1, 40, 64, 128]
        cost3 = self.classif3(out3) + cost2 # [bs, 1, 40, 64, 128]
        ###################
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdepth, left.size()[2], left.size()[3]], mode='trilinear') # [bs, 1, 80, 256, 512]
            cost2 = F.upsample(cost2, [self.maxdepth, left.size()[2], left.size()[3]], mode='trilinear') # [bs, 1, 80, 256, 512]

            cost1 = torch.squeeze(cost1, 1) # [bs, 80, 256, 512]
            pred1 = F.softmax(cost1, dim=1)
            pred1 = depthregression(self.maxdepth)(pred1) # [bs, 80, 256, 512] # sum*maxdepth

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = depthregression(self.maxdepth)(pred2)


        # during the testing phase, the final disp map is the last of three outputs
        cost3 = F.upsample(cost3, [self.maxdepth, left.size()[2], left.size()[3]], mode='trilinear') # [bs, 1, 80, 256, 512]
        cost3 = torch.squeeze(cost3, 1) # [4, 80, 256, 512]
        pred3 = F.softmax(cost3, dim=1) # [bs, 80, 256, 512]
        pred3_out = depthregression(self.maxdepth)(pred3) # [4, 256, 512]

        if self.training:
            return pred1, pred2, pred3_out
        else:
            return pred3_out
