import torch
import torch.nn as nn
from torch.autograd import Variable
from layers import upconvblock, convblock

# img_t = Variable(torch.randn(1, 3, 128, 416))
# img_tmin = Variable(torch.randn(1, 3, 128, 416))
# img_tplus = Variable(torch.randn(1, 3, 128, 416)

class PoseExpNet(nn.Module):

    def __init__(self, num_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.num_ref_imgs = num_ref_imgs
        self.output_exp = output_exp
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        upconv_planes = [256, 128, 64, 32, 16]

        self.conv1 = convblock(3 * (1 + self.num_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = convblock(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = convblock(conv_planes[1], conv_planes[2])
        self.conv4 = convblock(conv_planes[2], conv_planes[3])
        self.conv5 = convblock(conv_planes[3], conv_planes[4])
        self.conv6 = convblock(conv_planes[4], conv_planes[5])
        self.conv7 = convblock(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6 * self.num_ref_imgs, kernel_size=1, padding=0)

        self.upconv5 = upconvblock(conv_planes[4], upconv_planes[0])
        self.upconv4 = upconvblock(upconv_planes[0], upconv_planes[1])
        self.upconv3 = upconvblock(upconv_planes[1], upconv_planes[2])
        self.upconv2 = upconvblock(upconv_planes[2], upconv_planes[3])
        self.upconv1 = upconvblock(upconv_planes[3], upconv_planes[4])

        self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.num_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.num_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.num_ref_imgs, kernel_size=3, padding=1)
        self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.num_ref_imgs, kernel_size=3, padding=1)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.num_ref_imgs)
        x = [target_image]
        x.extend(ref_imgs)
        x = torch.cat(x, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.num_ref_imgs, 6)

        out_upconv5 = self.upconv5(out_conv5  )[:, :, 0: out_conv4.size(2), 0: out_conv4.size(3)]
        out_upconv4 = self.upconv4(out_upconv5)[:, :, 0: out_conv3.size(2), 0: out_conv3.size(3)]
        out_upconv3 = self.upconv3(out_upconv4)[:, :, 0: out_conv2.size(2), 0: out_conv2.size(3)]
        out_upconv2 = self.upconv2(out_upconv3)[:, :, 0: out_conv1.size(2), 0: out_conv1.size(3)]
        out_upconv1 = self.upconv1(out_upconv2)[:, :, 0: x.size(2), 0: x.size(3)]

        exp_mask4 = nn.functional.sigmoid(self.predict_mask4(out_upconv4))
        exp_mask3 = nn.functional.sigmoid(self.predict_mask3(out_upconv3))
        exp_mask2 = nn.functional.sigmoid(self.predict_mask2(out_upconv2))
        exp_mask1 = nn.functional.sigmoid(self.predict_mask1(out_upconv1))

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
