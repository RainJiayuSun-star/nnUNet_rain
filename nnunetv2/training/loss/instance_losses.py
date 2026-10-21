"""Two-foreground-label CC-DiceCE adaptation of Learning-to-Look-Closer.

CC scatter reduction adapted from TIO-IKIM/Learning-to-Look-Closer,
commit e5013bdb560eee544a94e1f068cb5cb4d003a51c, Apache-2.0 (see LICENSE).
Changes: core/edema one-vs-rest views, FP32 arithmetic, eager reduction, and
guarded geometry. No Blob losses or process-wide compiler settings are imported.
"""

import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.connected_components import compute_voronoi

UPSTREAM_REVISION = "e5013bdb560eee544a94e1f068cb5cb4d003a51c"


def binary_class_logits(logits: torch.Tensor, foreground: int) -> torch.Tensor:
    """Exact foreground-vs-rest logits; keep gradients into all three channels."""
    if logits.ndim != 5 or logits.shape[1] != 3 or foreground not in (1, 2):
        raise ValueError("Expected [B,3,D,H,W] logits and foreground label 1 or 2.")
    rest = [c for c in range(3) if c != foreground]
    return torch.stack((torch.logsumexp(logits[:, rest], dim=1), logits[:, foreground]), dim=1)


def binary_cc_dice_ce(logits: torch.Tensor, target: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Upstream binary reduction on supplied [B,D,H,W] Voronoi maps.

    Pure tensor arithmetic also runs on CPU for reference/parity tests. Production
    maps come from CUDA geometry. Average components per sample, then samples.
    ID 0 is reserved for an entirely empty foreground sample.
    """
    eps = 1e-7
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = logits.float().softmax(1)
        truth = target[:, 0].long()
        foreground = probabilities[:, 1]
        ce = -probabilities.clamp_min(eps).log().gather(1, truth[:, None]).squeeze(1)
        slots = max(1, int(components.max().item())) + 1
        losses = []
        for sample in range(logits.shape[0]):
            ids = components[sample].long().reshape(-1)

            def segment_sum(values, ids=ids):
                return torch.zeros(slots, device=logits.device, dtype=torch.float32).scatter_add(
                    0, ids, values.reshape(-1)
                )[1:]

            p = foreground[sample]
            t = (truth[sample] == 1).float()
            intersection = segment_sum(p * t)
            denominator = segment_sum(p) + segment_sum(t)
            count = segment_sum(torch.ones_like(p))
            per_component = 1 - 2 * intersection / denominator.clamp_min(eps)
            per_component = per_component + segment_sum(ce[sample]) / count.clamp_min(1)
            valid = (count > 0).float()
            cc = (per_component * valid).sum() / valid.sum().clamp_min(1)
            # The +1 preserves upstream's no-foreground Dice constant. CE still
            # penalizes foreground false positives; do not drop absent classes.
            losses.append(torch.where(valid.sum() > 0, cc, ce[sample].mean() + 1))
        return torch.stack(losses).mean()


class TwoLabelCCDiceCELoss(nn.Module):
    """Equal core/edema CC terms, optionally plus unsmoothed global Dice+CE."""

    def __init__(self, include_global: bool = False, batch_dice: bool = False, ddp: bool = False):
        super().__init__()
        self.global_loss = DC_and_CE_loss(
            {"batch_dice": batch_dice, "smooth": 0, "do_bg": False, "ddp": ddp},
            {}, weight_ce=1, weight_dice=1, ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        ) if include_global else None

    @torch.compiler.disable
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or logits.shape[1] != 3 or logits.shape[0] == 0:
            raise ValueError("CC trainers require nonempty [B,3,D,H,W] logits.")
        if target.shape != (logits.shape[0], 1, *logits.shape[2:]) or target.device != logits.device:
            raise ValueError("CC target must be [B,1,D,H,W] on the logits device.")
        if not torch.all((target == 0) | (target == 1) | (target == 2)):
            raise ValueError("CC target must contain only background=0, core=1, edema=2; no ignore labels.")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            terms = []
            for foreground in (1, 2):
                binary_target = (target == foreground).to(torch.int16)
                maps = torch.stack([compute_voronoi(sample[0]) for sample in binary_target])
                terms.append(binary_cc_dice_ce(binary_class_logits(logits, foreground), binary_target, maps))
            result = 0.5 * (terms[0] + terms[1])
            if self.global_loss is not None:
                result = result + self.global_loss(logits, target.long())
            return result
