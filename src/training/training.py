import os, sys
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR  # for lr schedule
import random
import numpy as np
import matplotlib.pyplot as plt


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))  # take the current dir
PARENT_DIRNAME = os.path.abspath(os.path.join(MY_DIRNAME, os.pardir))  # take the dir for needed import fold
sys.path.insert(0, PARENT_DIRNAME)
 
from utils import save_checkpoint, load_checkpoint_pretrain, load_checkpoint_resume
from utils import train, test# , inference_fdnet, train_foreground, test_foreground
from utils import inference_SLSFusion
from utils import func_utils
from utils import cuda_random_seed, setup_logger, parse
import models

from dataloader import dataloader_SLSFusion, myImageFloder_SLSFusion
from dataloader import SceneFlowLoader, listflowfile
from dataloader import SubmiteDataset

now = datetime.now()
today = now.strftime("%d-%m-%Y_%H-%M-%S")

def main(args):
    
    ### 
    ### SETUP LOG, ARGS ###
    ###
    
    cuda_random_seed(args)
    log = setup_logger(os.path.join(args.save_path + '/' + str(today) + '/logging', 'training.log'))  # /save/logging/
    log.info("************************ Start training ************************")
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))  # log all args infos
    writer = SummaryWriter(args.save_path + '/' + str(today) + '/tensorboardx')  # /save/tensorboardx

    ###
    ### DATA PREPARE & DATA LOADER ###
    ###
    
    if args.generate_depth_map:
        TrainImgLoader = None
        TestImgLoader = torch.utils.data.DataLoader(
            SubmiteDataset(args.datapath, args.data_list, args.dynamic_bs, separate=args.separate),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)
    elif args.dataset == 'kitti':
        train_data, val_data = dataloader_SLSFusion(args.datapath, args.split_train, args.split_val, args.depth_gth)
        TrainImgLoader = torch.utils.data.DataLoader(
            myImageFloder_SLSFusion(train_data, True, dynamic_bs=args.dynamic_bs, gth=args.depth_gth, separate=args.separate),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
        TestImgLoader = torch.utils.data.DataLoader(
            myImageFloder_SLSFusion(val_data, False, dynamic_bs=args.dynamic_bs, gth=args.depth_gth, separate=args.separate),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
    elif args.dataset == 'sceneflow':
        train_data, val_data = listflowfile.dataloader(args.datapath)
        TrainImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(train_data, True, calib=args.calib_value),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=False)
        TestImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(val_data, False, calib=args.calib_value),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)
    
    ###
    ### INITTIALIZE MODEL ###
    ###
    
    if args.data_type == 'depth':
        model = models.__dict__[args.arch](maxdepth=args.maxdepth, maxdisp=args.maxdisp, down=args.down)
    else:
        log.info('Model is not implemented')
        assert False

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = nn.DataParallel(model).cuda() # send all weights into gpu

    ###
    ### OPTIMIZER & SCHEDULER ###
    ###
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=args.lr_stepsize, gamma=args.lr_gamma)  # schedule for decay lr

    ###
    ### LOAD CHECKPOINT ###
    ###
    
    # this if for sceneflow into kitti
    if os.path.isfile(args.pretrain): # args.pretrain: name file
        model, checkpoint = load_checkpoint_pretrain(model, optimizer, scheduler, args, log) # (model, optimizer, scheduler, args, log):
    else:
        log.info('[Attention]: Can not find checkpoint {}'.format(args.pretrain))

    ###
    ### GENERATE DEPTH MAP ###
    ### 

    if args.generate_depth_map:
        if args.arch=='SLSFusion':
            os.makedirs(args.save_path  + '/' + str(today) + '/depth_maps/' + args.data_tag, exist_ok=True)
            tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
            for batch_idx, (imgL, imgR, left_4beams, right_4beams, left_mask, right_mask, calib, H, W, filename) in enumerate(tqdm_eval_loader):
                pred_disp = inference_SLSFusion(imgL, imgR, left_4beams, right_4beams, left_mask, right_mask, None, calib, None, model, None)
                for idx, name in enumerate(filename):
                    np.save(args.save_path + '/' + str(today) + '/depth_maps/' + args.data_tag + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])
            import sys
            sys.exit()

    ###
    ### TRAINING/ EVALUATION ###
    ###
    
    best_RMSE = 1e10
    loss_train_plot = []
    loss_val_plot = []
    plt.ion()  # for update plot
    fig = plt.figure(1)
    
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        
        scheduler.step() # decay learning rate
        is_best = False
        
        phase_set = ['train', 'validation'] if (epoch % args.eval_interval) == 0 else ['train'] # calcul validation for certain set
        for phase in phase_set:
            metric = func_utils.Metric()
            running_loss = 0.0
            dataloader = TrainImgLoader if phase == 'train' else TestImgLoader
            tqdm_dataloader = tqdm(dataloader, total=len(dataloader))
            for batch_idx, input_data in enumerate(tqdm_dataloader):
                if args.dataset=='sceneflow':
                    calib = input_data[-1]
                    depth = input_data[-2]
                    input_data = input_data[:2]
                    bs = input_data[0].size()[0]
                    h = input_data[0].size()[2]
                    w = input_data[0].size()[3]
                    left_4beams = torch.zeros([bs, 1, h, w])
                    right_4beams = torch.zeros([bs, 1, h, w])
                    left_mask = torch.zeros([bs, 1, h, w])
                    right_mask = torch.zeros([bs, 1, h, w])
                    input_data.extend([left_4beams, right_4beams, left_mask, right_mask, depth, calib])

                input_data.extend([metric, model, optimizer])
                train(*input_data) if phase == 'train' else test(*input_data)

                tqdm_dataloader.set_description("loss: %s" % str(float(metric.RMSELIs.avg)))
                running_loss += metric.RMSELIs.avg

            log.info(metric.print(0, '{}'.format('TRAIN Epoch ' if phase == 'train' else 'TEST Epoch') + str(epoch+1)))
            metric.tensorboard(writer, epoch, token='TRAIN' if phase == 'train' else 'TEST')

            epoch_loss = float(running_loss / (batch_idx + 1))
            loss_train_plot.append(epoch_loss) if phase == 'train' else loss_val_plot.append(epoch_loss)

            if phase == 'validation':
                loss_RMSE = int(metric.RMSELIs.avg)
                is_best = loss_RMSE < best_RMSE
                if is_best==True:
                    best_RMSE = loss_RMSE

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_RMSE': best_RMSE,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch, args, log, folder=args.save_path + '/' + str(today))
            log.info("Save checkpoint with is_best: {}, epoch: {}".format(is_best, epoch+1))

        log.info('Loss_train_plot: {}'.format(loss_train_plot))
        log.info('Loss_val_plot: {}'.format(loss_val_plot))
        plt.plot(loss_train_plot)
        plt.plot(loss_val_plot)
        plt.draw()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.title('Model Loss')
        plt.pause(0.01)
        fig.savefig(args.save_path + '/' + str(today) + '/logging' + '/training_' + args.arch + '.png')
        fig.savefig(args.save_path + '/' + str(today) + '/logging' + '/training_' + args.arch + '.svg')
        if epoch != (args.epochs - 1):
            plt.clf()

if __name__ == '__main__':
    args = parse()
    main(args=args)