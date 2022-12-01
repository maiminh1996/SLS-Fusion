import numpy as np

def absolute_relative_error(gt, pred):
    # Mean Absolute Relative Error absErrorRel
    num_pixel = gt.shape[0]
    diff = np.abs(gt - pred) / gt  # compute errors
    absErrorRel = np.sum(diff)/ num_pixel
    return absErrorRel

def square_relative_error(gt, pred):
    # Mean square Relative Error sqErrorRel
    num_pixel = gt.shape[0]
    diff = (np.abs(gt - pred))**2 / gt  # compute errors
    sqErrorRel = np.sum(diff)/ num_pixel
    return sqErrorRel

def Square_Mean_Relative_Error(gt, pred):
    # Square Mean Relative Error
    scale = 10.0
    # scale = 1
    gt_scale = gt*scale
    pred_scale = pred*scale
    s_rel = ((gt_scale - pred_scale) * (gt_scale - pred_scale)) / (gt_scale * gt_scale)  # compute errors
    # squa_rel_sum = np.sum(s_rel)
    # print(s_rel.shape)

    squa_rel_sum = np.mean(s_rel)
    return squa_rel_sum

# def Root_Mean_Square_error(gt, pred):
#     # Root Mean Square error RMSE
#     scale = 10.0
#     gt_scale = gt * scale
#     scale = 10.0
#     pred_scale = pred * scale
#     square = (gt_scale - pred_scale) ** 2
#     rms_squa_sum = np.sum(square)
#     return rms_squa_sum


def Log_Root_Mean_Square_error(gt, pred):
    # Log Root Mean Square error
    scale = 10.0
    gt_scale = gt * scale
    pred_scale = pred * scale
    log_square = (np.log(gt_scale) - np.log(pred_scale)) ** 2
    log_rms_sum = np.sum(log_square)
    return log_rms_sum

def Scale_invariant_error(gt, pred):
    # Scale invariant error
    scale = 10.0
    gt_scale = gt * scale
    pred_scale = pred * scale
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)

    diff_log_2 = diff_log ** 2
    diff_log_2_sum = np.sum(diff_log_2)
    return diff_log_sum, diff_log_2_sum

# def scale_invariant_error_log(gt, pred):
#     # Scale invariant error SILog
#     diff_log = np.log(gt) - np.log(pred) + alpha
#     # diff_log_sum = np.sum(diff_log)

#     diff_log_2 = diff_log ** 2
#     diff_log_2_sum = np.sum(diff_log_2)
#     return diff_log_sum, diff_log_2_sum

def scale_invariant_error_log(gt, pred):
    # Scale invariant logarithmic error SILog
    num_pixel = gt.shape[0]
    di = np.log(gt) - np.log(pred) # + alpha
    SILog = np.sum((di**2))/ num_pixel - ((np.sum(di))**2)/ (num_pixel**2)
    return SILog

def Mean_log10_error(gt, pred):
    # Mean log10 error
    log10_sum = np.sum(np.abs(np.log10(gt) - np.log10(pred)))
    return log10_sum

def root_mean_spuare_error(gt, pred):
    """RMSE"""
    num_pixel = gt.shape[0]
    # diff = (gt - pred)*(gt - pred)
    diff = (gt - pred)**2
    diff = np.sum(diff)/ num_pixel
    rmse = diff**(0.5)
    return rmse

def inverse_root_mean_spuare_error(gt, pred):
    """iRMSE"""
    i_gt = 1/ gt
    i_pred = 1/ pred
    num_pixel = gt.shape[0]
    diff = (i_gt - i_pred)**2
    diff = np.sum(diff)/ num_pixel
    irmse = diff**(0.5)
    return irmse

def mean_absolute_error(gt, pred):
    """MAE"""
    num_pixel = gt.shape[0]
    diff = np.abs(gt - pred)  # compute errors
    mae = np.sum(diff)/ num_pixel
    return mae

def inverse_mean_absolute_error(gt, pred):
    """iMAE"""
    i_gt = 1/ gt
    i_pred = 1/ pred
    num_pixel = gt.shape[0]
    diff = np.abs(i_gt - i_pred)  # compute errors
    imae = np.sum(diff)/ num_pixel
    return imae