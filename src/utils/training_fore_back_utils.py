import os
import shutil
import torch
import torch.nn.functional as F

from loss import L1_loss

def train_foreground(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, foreground_mask, depth, calib, metric_log, model, optimizer):
    model.train()
    calib = calib.float()

    imgL, imgR, depth, calib = imgL.cuda(), imgR.cuda(), depth.cuda(), calib.cuda()
    foreground_mask = foreground_mask.cuda()
    sparseL, sparseR, maskL, maskR = sparse_left.cuda(), sparse_right.cuda(), mask_left.cuda(), mask_right.cuda()

    mask = (depth >= 1) * (depth <= 80)

    foreground_Mask = (depth >= 1) * (depth <= 80) * (foreground_mask>0)
    background_Mask = (depth >= 1) * (depth <= 80) * (foreground_mask<1)

    foreground_Mask.detach_()
    background_Mask.detach_()
    mask.detach_()  # contain pixel with gth

    optimizer.zero_grad()

    try:
        output1, output2, output3 = model(imgL, imgR, sparseL, sparseR, maskL, maskR, calib)  # [bs, 256, 512]
        # print('FDNet')
    except TypeError:
        # print("SDNet")
        output1, output2, output3 = model(imgL, imgR, calib)  # [bs, 256, 512]

    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)

    """
    # TODO
    if args.data_type == 'disparity':
        output1 = disp2depth(output1, calib)
        output2 = disp2depth(output2, calib)
        output3 = disp2depth(output3, calib)
    """
    loss= 0.5 * F.smooth_l1_loss(output1[foreground_Mask], depth[foreground_Mask], size_average=True) + 0.7 * F.smooth_l1_loss(
        output2[foreground_Mask], depth[foreground_Mask], size_average=True) + F.smooth_l1_loss(output3[foreground_Mask], depth[foreground_Mask],
                                                                          size_average=True)

    print("loss: ", 0.5 * F.smooth_l1_loss(output1[foreground_Mask], depth[foreground_Mask], size_average=True) + 0.7 * F.smooth_l1_loss(
        output2[foreground_Mask], depth[foreground_Mask], size_average=True) + F.smooth_l1_loss(output3[foreground_Mask], depth[foreground_Mask],
                                                                          size_average=True))
    metric_log.calculate(depth, output3, loss=loss.item())
    loss.backward()
    optimizer.step()


def test_foreground(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, foreground_mask, depth, calib, metric_log, model, optimizer):
    model.eval()
    calib = calib.float()
    imgL, imgR, calib, depth = imgL.cuda(), imgR.cuda(), calib.cuda(), depth.cuda()
    # foreground_mask = foreground_mask.cuda()
    sparseL, sparseR, maskL, maskR = sparse_left.cuda(), sparse_right.cuda(), mask_left.cuda(), mask_right.cuda()

    mask = (depth >= 1) * (depth <= 80)
    mask.detach_()
    # foreground_Mask = (depth >= 1) * (depth <= 80) * (foreground_mask>0)
    # foreground_Mask.detach_()
    with torch.no_grad():
        # output3 = model(imgL, imgR, calib)
        try:
            output3 = model(imgL, imgR, sparseL, sparseR, maskL, maskR, calib)
            # print('FDNet')
        except TypeError:
            # print("SDNet")
            output3 = model(imgL, imgR, calib)

        # output3 = model(imgL, imgR, sparseL, sparseR, maskL, maskR, calib)
        output3 = torch.squeeze(output3, 1)
        
        """
        # TODO
        if args.data_type == 'disparity':
            output3 = disp2depth(output3, calib)
        """

        loss = F.smooth_l1_loss(output3[mask], depth[mask], size_average=True)
        # loss = F.smooth_l1_loss(output3[foreground_Mask], depth[foreground_Mask], size_average=True)

        metric_log.calculate(depth, output3, loss=loss.item())

    torch.cuda.empty_cache()
    return