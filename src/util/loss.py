# Author: Bingxin Ke
# Last modified: 2024-02-22
import torch.nn.functional as F
import pytorch_msssim
import torch
from src.util.HDN_demo import HDN_interface


def get_loss(loss_name, **kwargs):
    if "silog_mse" == loss_name:
        criterion = SILogMSELoss(**kwargs)
    elif "silog_rmse" == loss_name:
        criterion = SILogRMSELoss(**kwargs)
    elif "mse_loss" == loss_name:
        criterion = torch.nn.MSELoss(**kwargs)
    elif "l1_loss" == loss_name:
        criterion = torch.nn.L1Loss(**kwargs)
    elif "l1_loss_with_mask" == loss_name:
        criterion = L1LossWithMask(**kwargs)
    elif "mean_abs_rel" == loss_name:
        criterion = MeanAbsRelLoss()
    elif "mae_mse_ssim_loss" == loss_name:
        criterion = MaeMseSsimLoss(**kwargs)
    elif "HDNLoss" == loss_name:
        criterion = HDN_interface(**kwargs)
    elif "high_punish_mse_loss" == loss_name:
        criterion = HighPunishMseLoss(**kwargs)
    else:
        raise NotImplementedError

    return criterion


class HighPunishMseLoss:
    def __init__(self, mse_weight=1.0, high_punish_weight=2.0):
        self.mse_weight = mse_weight  # 原始的MSE权重
        self.high_punish_weight = high_punish_weight  # 对于pred > target的惩罚权重

    def __call__(self, pred, target):
        # 计算常规的 MSE loss
        mse_loss = F.mse_loss(pred, target)

        # 计算当pred > target时的惩罚项
        high_punish_mask = (pred > target).float()  # pred大于target的地方
        high_punish_loss = F.mse_loss(pred * high_punish_mask, target * high_punish_mask)

        # 总损失 = 正常MSE损失 + 高惩罚损失
        total_loss = self.mse_weight * mse_loss + self.high_punish_weight * high_punish_loss
        return total_loss


class MaeMseSsimLoss:
    def __init__(self, mae_weight=0.3, mse_weight=0.2, ssim_weight=0.5, data_range=1, channel=3):
        self.alpha = mae_weight
        self.beta = mse_weight
        self.gamma = ssim_weight
        self.ssim_loss = pytorch_msssim.SSIM(data_range=data_range, size_average=True, channel=channel)

    def __call__(self, pred, target):
        # print(f"pred min: {pred.min()}, pred max: {pred.max()}, target min: {target.min()}, target max: {target.max()}")
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)

        # **对整个张量计算 SSIM**
        ssim_loss = 1 - self.ssim_loss(pred, target)

        total_loss = self.alpha * l1_loss + self.beta * l2_loss + self.gamma * ssim_loss
        # total_loss = self.alpha * l1_loss + self.beta * l2_loss
        return total_loss


class L1LossWithMask:
    def __init__(self, batch_reduction=False):
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        diff = depth_pred - depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        loss = torch.sum(torch.abs(diff)) / n
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class MeanAbsRelLoss:
    def __init__(self) -> None:
        # super().__init__()
        pass

    def __call__(self, pred, gt):
        diff = pred - gt
        rel_abs = torch.abs(diff / gt)
        loss = torch.mean(rel_abs, dim=0)
        return loss


class SILogMSELoss:
    def __init__(self, lamb, log_pred=True, batch_reduction=True):
        """Scale Invariant Log MSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogMSELoss, self).__init__()
        self.lamb = lamb
        self.pred_in_log = log_pred
        self.batch_reduction = batch_reduction

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = (
            depth_pred if self.pred_in_log else torch.log(torch.clip(depth_pred, 1e-8))
        )
        log_depth_gt = torch.log(torch.clip(depth_gt, 1e-8))

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)

        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n ** 2)
        loss = first_term - second_term
        if self.batch_reduction:
            loss = loss.mean()
        return loss


class SILogRMSELoss:
    def __init__(self, lamb, alpha, log_pred=True):
        """Scale Invariant Log RMSE Loss

        Args:
            lamb (_type_): lambda, lambda=1 -> scale invariant, lambda=0 -> L2 loss
            alpha:
            log_pred (bool, optional): True if model prediction is logarithmic depht. Will not do log for depth_pred
        """
        super(SILogRMSELoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.pred_in_log = log_pred

    def __call__(self, depth_pred, depth_gt, valid_mask=None):
        log_depth_pred = depth_pred if self.pred_in_log else torch.log(depth_pred)
        log_depth_gt = torch.log(torch.clip(depth_gt, 1e-8))
        # borrowed from https://github.com/aliyun/NeWCRFs
        # diff = log_depth_pred[valid_mask] - log_depth_gt[valid_mask]
        # return torch.sqrt((diff ** 2).mean() - self.lamb * (diff.mean() ** 2)) * self.alpha

        diff = log_depth_pred - log_depth_gt
        if valid_mask is not None:
            diff[~valid_mask] = 0
            n = valid_mask.sum((-1, -2))
        else:
            n = depth_gt.shape[-2] * depth_gt.shape[-1]

        diff2 = torch.pow(diff, 2)
        first_term = torch.sum(diff2, (-1, -2)) / n
        second_term = self.lamb * torch.pow(torch.sum(diff, (-1, -2)), 2) / (n ** 2)
        loss = torch.sqrt(first_term - second_term).mean() * self.alpha
        return loss
