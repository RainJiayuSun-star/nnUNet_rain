from typing import Callable
import torch
from torch import nn
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalTverskyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33,
                 batch_dice: bool = False, do_bg: bool = True, smooth: float = 1e-5, ddp: bool = True):
        """
        Focal Tversky Loss implementation for nnU-Net v2.
        
        Tversky Index (TI):
            TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
            
        Focal Tversky Loss (FTL):
            FTL = (1 - TI) ** gamma
            
        Parameters:
            alpha: Weight for False Positives (FP).
            beta: Weight for False Negatives (FN). Setting beta > alpha (e.g. alpha=0.3, beta=0.7)
                  penalizes missed targets more heavily, boosting small lesion recall.
            gamma: Focal exponent focusing loss on hard (low TI) samples.
            batch_dice: If True, computes Tversky index globally across batch samples.
            do_bg: If True, includes background class in loss computation.
            smooth: Smoothing constant to avoid division by zero.
            ddp: Distributed Data Parallel flag for gradient gather.
        """
        super(FocalTverskyLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # ground truth is already one-hot encoded
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        # Calculate Tversky Index: TP / (TP + alpha * FP + beta * FN)
        # FP = sum_pred - intersect, FN = sum_gt - intersect
        denominator = (1.0 - self.alpha - self.beta) * intersect + self.alpha * sum_pred + self.beta * sum_gt + float(self.smooth)
        tversky_index = (intersect + float(self.smooth)) / denominator.clamp_min(1e-8)

        # Focal Tversky Loss: (1 - TI)^gamma
        focal_tversky_loss = torch.pow((1.0 - tversky_index).clamp_min(1e-8), self.gamma)

        return focal_tversky_loss.mean()


class FTL_and_CE_loss(nn.Module):
    def __init__(self, ftl_kwargs, ce_kwargs, weight_ce=1.0, weight_ftl=1.0, ignore_label=None):
        """
        Compound Loss combining Focal Tversky Loss and Robust Cross-Entropy Loss.
        """
        super(FTL_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ftl = weight_ftl
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ftl = FocalTverskyLoss(apply_nonlin=softmax_helper_dim1, **ftl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        ftl_loss = self.ftl(net_output, target_dice, loss_mask=mask) if self.weight_ftl != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        return self.weight_ce * ce_loss + self.weight_ftl * ftl_loss


class FTL_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, ftl_kwargs, weight_ce=1.0, weight_ftl=1.0, use_ignore_label: bool = False):
        """
        Compound Loss combining Focal Tversky Loss and Binary Cross-Entropy Loss (for region-based targets).
        """
        super(FTL_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_ftl = weight_ftl
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.ftl = FocalTverskyLoss(apply_nonlin=torch.sigmoid, **ftl_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        ftl_loss = self.ftl(net_output, target_regions, loss_mask=mask) if self.weight_ftl != 0 else 0
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        return self.weight_ce * ce_loss + self.weight_ftl * ftl_loss
