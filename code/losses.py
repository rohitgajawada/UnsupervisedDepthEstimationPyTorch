from __future__ import division
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, 
                                    explainability_mask, pose, rotation_mode='euler',
                                    padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h,w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h,w)) for ref_img in ref_imgs]

        intrinsics_scaled = torch.cat((intrinsics[:, 0,2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:,:,0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for i , ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * out_of_bound

            if explainability_mask is not None:
                diff = diff * explainability_mask[:, i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()

            # assert((reconstruction_loss == reconstruction_loss).data[0] == 1) # WTF is this
        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]

    if type(depth) not in [tuple, list]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss += one_scale(d, mask)
    return loss


def explainability_loss(mask):
    """
    Regularizer for the explainability mask. Forces non-zero mask.
    @param mask: list/tuple of explainability masks, at different scales.
    """
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        # Requires mask to be an autograd Variable
        ones = Variable(torch.ones(mask_scaled.size()))
        loss += nn.functional.binary_cross_entropy(mask, ones)
    return loss


def gradient(pred):
    """
    Returns the gradient as a tuple
    @param pred: 3D tensor
    """
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


def smooth_loss(pred):
    """
    Multi scale smoothing loss for the image
    @param pred: list/tuple of predictions at different scales
    """
    if type(pred) not in [list, tuple]:
        pred = [pred]

    loss = 0
    weight = 1

    for pred_scaled in pred:
        dx, dy = gradient(pred_scaled)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean()
                + dydx.abs().mean() + dy2.abs().mean()) * weight
        weight /= 2.83 # 2sqrt(2)
    return loss


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    Crop used by Garg ECCV16 to reproduce Eigen NIPS 2014 results
    Constructs a mask of False values, the same size as target
    and then set to True inside the crop.
    Seems a lot of magic numbers are used.
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) and (current_gt < 80)
        if crop:
            valid = valid and crop_mask

        valid_gt = curren_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt) / torch.mediam(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (threst < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred) ** 2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1,a2,a3]]
