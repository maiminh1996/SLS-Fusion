import torch.nn.functional as F

def L1_loss(pred, gth):
    loss = F.smooth_l1_loss(pred, gth, size_average=True)
    return loss


