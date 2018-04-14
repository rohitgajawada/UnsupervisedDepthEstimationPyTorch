import torch
import torch.nn as nn
from torch.autograd import Variable
# from models.layers import downsample_convblock, upconvblock, convblock, predict_disp, crop_like

# img_t = Variable(torch.randn(1, 3, 128, 416))
# img_tmin = Variable(torch.randn(1, 3, 128, 416))
# img_tplus = Variable(torch.randn(1, 3, 128, 416))


def downsample_convblock(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.ReLU(inplace=True)
            )


def predict_disp(in_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


def convblock(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


def upconvblock(in_planes, out_planes):
    return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]

        self.conv1 = downsample_convblock(3, conv_planes[0], kernel_size=7)
        self.conv2 = downsample_convblock(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_convblock(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_convblock(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_convblock(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_convblock(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_convblock(conv_planes[5], conv_planes[6])

        self.upconv7 = upconvblock(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconvblock(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconvblock(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconvblock(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconvblock(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconvblock(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconvblock(upconv_planes[5], upconv_planes[6])

        self.iconv7 = convblock(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = convblock(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = convblock(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = convblock(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = convblock(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = convblock(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = convblock(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        #print(out_conv6.size())
        out_conv7 = self.conv7(out_conv6)
        #print(out_conv7.size())

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(nn.functional.upsample(disp4, scale_factor=2, mode='bilinear'), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(nn.functional.upsample(disp3, scale_factor=2, mode='bilinear'), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(nn.functional.upsample(disp2, scale_factor=2, mode='bilinear'), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1
