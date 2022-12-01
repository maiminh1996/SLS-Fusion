import os
import shutil
import torch
import torch.nn.functional as F

from loss import L1_loss

def save_checkpoint(state, is_best, epoch, args, log, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename) # latest checkpoint
    # shutil.move(folder + '/' + filename, '/media/zelt/Données/KITTI/test_algo/save_weight_FDNet/results/sdn_kitti_train_set' + '/' + filename)
    log.info("Save weight: {}".format(folder + '/' + filename))
    if is_best: # best
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')
        # shutil.move(folder + '/model_best.pth.tar', '/media/zelt/Données/KITTI/test_algo/save_weight_FDNet/results/sdn_kitti_train_set' + '/model_best.pth.tar')
        log.info("Copy weight to: {}".format(folder + '/model_best.pth.tar'))
    if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0: # each
        shutil.copyfile(folder + '/' + filename, folder + '/checkpoint_{}.pth.tar'.format(epoch + 1))
        # shutil.move(folder + '/checkpoint_{}.pth.tar'.format(epoch + 1), '/media/zelt/Données/KITTI/test_algo/save_weight_FDNet/results/sdn_kitti_train_set' + '/checkpoint_{}.pth.tar'.format(epoch + 1))
        log.info("Copy weight to: {}".format(folder + '/checkpoint_{}.pth.tar'.format(epoch + 1)))

def load_checkpoint_pretrain(model, optimizer, scheduler, args, log):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.

    log.info("=> loading pretrain '{}'".format(args.pretrain))
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint

def load_checkpoint_resume(model, optimizer, scheduler, filename, log):
    # start_epoch = 0
    log.info("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_RMSE = checkpoint['best_RMSE']
    scheduler.load_state_dict(checkpoint['scheduler'])
    log.info("=> loaded checkpoint '{}' (epoch {})"
             .format(filename, checkpoint['epoch']))
    return model, optimizer, start_epoch, checkpoint, best_RMSE, scheduler

def train(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, depth, calib, metric_log, model, optimizer):
    model.train()
    calib = calib.float()

    imgL, imgR, depth, calib = imgL.cuda(), imgR.cuda(), depth.cuda(), calib.cuda()
    sparseL, sparseR, maskL, maskR = sparse_left.cuda(), sparse_right.cuda(), mask_left.cuda(), mask_right.cuda()
    # [12, 3, 256, 512], [12, 3, 256, 512], [12, 256, 512], [12]

    mask = (depth >= 1) * (depth <= 80)
    mask.detach_() # contain pixel with gth

    optimizer.zero_grad()

    # SLSFusion
    output1, output2, output3 = model(imgL, imgR, sparseL, sparseR, maskL, maskR, calib)  # [bs, 256, 512]
    # SDNet
    # output1, output2, output3 = model(imgL, imgR, calib)  # [bs, 256, 512]

    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)

    loss = 0.5*L1_loss(output1[mask], depth[mask]) \
        + 0.7*L1_loss(output2[mask], depth[mask]) \
            + L1_loss(output3[mask], depth[mask])
    
    metric_log.calculate(depth, output3, loss=loss.item())
    loss.backward()
    optimizer.step()

def test(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, depth, calib, metric_log, model, optimizer):
    model.eval()
    calib = calib.float()
    imgL, imgR, calib, depth = imgL.cuda(), imgR.cuda(), calib.cuda(), depth.cuda()
    sparseL, sparseR, maskL, maskR = sparse_left.cuda(), sparse_right.cuda(), mask_left.cuda(), mask_right.cuda()

    mask = (depth >= 1) * (depth <= 80)
    mask.detach_()

    with torch.no_grad():
        # SLSFusion output
        output3 = model(imgL, imgR, sparseL, sparseR, maskL, maskR, calib)
        output3 = torch.squeeze(output3, 1)
        # Loss
        loss = L1_loss(output3[mask], depth[mask])
        # Metric
        metric_log.calculate(depth, output3, loss=loss.item())
    torch.cuda.empty_cache()

def inference_SLSFusion(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, depth, calib, metric_log, model, optimizer):
    model.eval()
    imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, calib = imgL.cuda(), imgR.cuda(), sparse_left.cuda(), sparse_right.cuda(), mask_left.cuda(), mask_right.cuda(), calib.float().cuda()


    with torch.no_grad():
        output3 = model(imgL, imgR, sparse_left, sparse_right, mask_left, mask_right, calib)  # [bs, 256, 512]
        # if args.data_type == 'disparity':
        #     output = disp2depth(output, calib)
    pred_disp = output3.data.cpu().numpy()

    return pred_disp

def inference_sdnet(imgL, imgR, calib, model):
    model.eval()
    imgL, imgR, calib = imgL.cuda(), imgR.cuda(), calib.float().cuda()

    with torch.no_grad():
        output = model(imgL, imgR, calib)
    pred_disp = output.data.cpu().numpy()

    return pred_disp



