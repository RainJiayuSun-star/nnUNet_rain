"""CPU mathematical checks; SciPy is a test oracle, never a training fallback."""

import ast
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt, label

from nnunetv2.training.loss import instance_losses as cc
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def reference_voronoi(mask):
    components, count = label(mask.cpu().numpy() > 0, structure=np.ones((3, 3, 3)))
    if count:
        indices = distance_transform_edt(components == 0, return_distances=False, return_indices=True)
        components = components[tuple(indices)]
    return torch.as_tensor(components, device=mask.device, dtype=torch.int64)


def explicit_loss(logits, target, maps):
    p = logits.float().softmax(1)
    ce = -p.clamp_min(1e-7).log().gather(1, target.long()).squeeze(1)
    samples = []
    for b in range(len(p)):
        components = []
        for k in maps[b].unique().tolist():
            if k == 0:
                continue
            region = maps[b] == k
            pred = p[b, 1][region]
            truth = target[b, 0][region].float()
            components.append(1 - 2 * (pred * truth).sum() / (pred.sum() + truth.sum()).clamp_min(1e-7)
                              + ce[b][region].mean())
        samples.append(torch.stack(components).mean() if components else ce[b].mean() + 1)
    return torch.stack(samples).mean()


def sample_target():
    target = torch.zeros(2, 1, 7, 7, 7, dtype=torch.int16)
    target[0, 0, 1, 1, 1] = 1
    target[0, 0, 4:6, 4:6, 4:6] = 1
    # Sample 1 has no metastasis.
    return target


def test_vectorized_value_and_gradient_match_explicit():
    torch.manual_seed(2)
    y = sample_target()
    maps = torch.stack([reference_voronoi(s[0]) for s in y])
    logits = torch.randn(2, 2, 7, 7, 7, requires_grad=True)
    actual = cc.binary_cc_dice_ce(logits, y, maps)
    expected = explicit_loss(logits, y, maps)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_upstream_reduction_parity():
    root = os.environ.get("CC_UPSTREAM_ROOT")
    if root is None:
        reference = Path("lossFunctions/ccDice/Learning-to-Look-Closer")
        root = next((parent / reference for parent in Path(__file__).resolve().parents
                     if (parent / reference).is_dir()), None)
    if root is None:
        pytest.skip("Set CC_UPSTREAM_ROOT to the pinned upstream checkout for parity testing.")
    upstream = Path(root) / "nnunetv2/training/loss/instance_losses.py"
    if not upstream.is_file():
        pytest.skip("Set CC_UPSTREAM_ROOT to the pinned upstream checkout for parity testing.")
    # Execute only the original arithmetic functions, not upstream imports,
    # CuPy loading, compilation, or process-wide Dynamo configuration mutations.
    names = {"_scatter_segment_sum", "_combine_vectorized_loss_terms", "_whole_volume_fallback_loss",
             "_prepare_vectorized_loss_inputs", "_vectorized_cc_reduction"}
    pinned = subprocess.run(
        ["git", "-C", str(upstream.parents[3]), "show",
         f"{cc.UPSTREAM_REVISION}:nnunetv2/training/loss/instance_losses.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    tree = ast.parse(pinned)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(definitions) == len(names)
    namespace = {"torch": torch, "vmap": torch.func.vmap, "Optional": Optional,
                 "Callable": Callable, "VectorizedLossMode": str}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(upstream), "exec"), namespace)  # noqa: S102 -- selected reference functions
    y = (sample_target() == 1).short()
    maps = torch.stack([reference_voronoi(s[0]) for s in y])
    logits = torch.randn(2, 2, 7, 7, 7, requires_grad=True)
    pred, truth, ce, slots = namespace["_prepare_vectorized_loss_inputs"](logits, y, maps, lambda x: x.softmax(1))
    expected = namespace["_vectorized_cc_reduction"](pred, truth, ce, maps, "dice_ce", slots)
    actual = cc.binary_cc_dice_ce(logits, y, maps)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_exact_global_addition(monkeypatch):
    monkeypatch.setattr(cc, "compute_voronoi", reference_voronoi)
    logits = torch.randn(2, 2, 7, 7, 7, requires_grad=True)
    y = sample_target()
    instance = cc.BinaryCCDiceCELoss()
    combined = cc.BinaryCCDiceCELoss(include_global=True)
    maps = torch.stack([reference_voronoi(s[0]) for s in y])
    expected = explicit_loss(logits, y, maps)
    torch.testing.assert_close(instance(logits, y), expected)
    torch.testing.assert_close(combined(logits, y), expected + combined.global_loss(logits, y.long()))
    assert combined.global_loss.dc.smooth == 0
    assert not combined.global_loss.dc.do_bg


def test_global_fp32_under_autocast():
    logits = torch.randn(2, 2, 7, 7, 7, dtype=torch.bfloat16, requires_grad=True)
    y = sample_target()
    loss = cc.global_dice_ce()
    expected = loss(logits.float(), y)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(logits, y)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("batch_dice", [False, True])
def test_global_value_and_gradient_match_explicit(batch_dice):
    y = sample_target()
    logits = torch.randn(2, 2, 7, 7, 7, requires_grad=True)
    p, truth = logits.softmax(1)[:, 1], y[:, 0].float()
    axes = (0, 1, 2, 3) if batch_dice else (1, 2, 3)
    dice = (2 * (p * truth).sum(axes) / (p.sum(axes) + truth.sum(axes)).clamp_min(1e-8)).mean()
    expected = torch.nn.functional.cross_entropy(logits, y[:, 0].long()) - dice
    actual = cc.global_dice_ce(batch_dice=batch_dice)(logits, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("kind", ["empty", "full", "tiny", "gap", "bridge", "fragment"])
def test_edge_cases_have_finite_gradients(monkeypatch, kind):
    monkeypatch.setattr(cc, "compute_voronoi", reference_voronoi)
    y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.int16)
    if kind == "full":
        y.fill_(1)
    elif kind != "empty":
        y[0, 0, 1, 1, 1] = 1
        y[0, 0, 5, 1, 1] = 1
        if kind != "tiny":
            y[0, 0, 1:6, 3:5, 1:3] = 1
    logits = torch.randn(1, 2, 7, 7, 7, requires_grad=True)
    # Deliberately different topology in the prediction. No topology guarantee
    # is asserted: the actual acceptance criterion is valid loss/gradients.
    with torch.no_grad():
        if kind == "bridge":
            logits[:, 1, 1:6, 1, 1] = 4
        elif kind == "fragment":
            logits[:, 1, 3, 3:5, 1:3] = -4
        elif kind == "gap":
            logits[:, 0, :, 2, :] = 4
    loss = cc.BinaryCCDiceCELoss(True)(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_empty_class_false_positive_penalty():
    y = torch.zeros(1, 1, 3, 3, 3, dtype=torch.int16)
    maps = torch.zeros(1, 3, 3, 3, dtype=torch.int64)
    logits = torch.zeros(1, 2, 3, 3, 3, requires_grad=True)
    loss = cc.binary_cc_dice_ce(logits, y, maps)
    torch.testing.assert_close(loss, torch.tensor(1 + np.log(2), dtype=torch.float32))
    loss.backward()
    assert (logits.grad[:, 1] > 0).all()


def test_missing_small_lesion_and_false_positive_increase_loss():
    y = (sample_target()[:1] == 1).short()
    maps = torch.stack([reference_voronoi(s[0]) for s in y])
    logits = torch.cat([(1 - y.float()) * 8, y.float() * 8], dim=1)
    good = cc.binary_cc_dice_ce(logits, y, maps)
    missing = logits.clone()
    missing[:, 0, 1, 1, 1], missing[:, 1, 1, 1, 1] = 8, 0
    fp = logits.clone()
    fp[:, 0, 0, 6, 0], fp[:, 1, 0, 6, 0] = 0, 8
    assert cc.binary_cc_dice_ce(missing, y, maps) > good
    assert cc.binary_cc_dice_ce(fp, y, maps) > good


def test_deep_supervision_recomputes_and_skips_zero_weights(monkeypatch):
    calls = []

    def geometry(mask):
        calls.append(tuple(mask.shape))
        return reference_voronoi(mask)

    monkeypatch.setattr(cc, "compute_voronoi", geometry)
    outputs = [torch.randn(1, 2, n, n, n, requires_grad=True) for n in (8, 4, 2)]
    targets = [torch.zeros(1, 1, n, n, n, dtype=torch.int16) for n in (8, 4, 2)]
    targets[0][0, 0, 1, 1, 1] = 1  # Absent at coarse resolutions.
    loss = DeepSupervisionWrapper(cc.BinaryCCDiceCELoss(), [2/3, 1/3, 0])
    actual = loss(outputs, targets)
    assert calls == [(8, 8, 8), (4, 4, 4)]
    expected = sum(w * loss.loss(x, y) for w, x, y in zip([2/3, 1/3], outputs, targets))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert outputs[0].grad.abs().sum() > 0 and outputs[1].grad.abs().sum() > 0
    assert outputs[2].grad is None


def test_invalid_target_and_cpu_geometry_fail():
    with pytest.raises(ValueError, match="only background"):
        cc.BinaryCCDiceCELoss()(torch.zeros(1, 2, 2, 2, 2), torch.full((1, 1, 2, 2, 2), 2))
    with pytest.raises(ValueError, match="CUDA"):
        cc.BinaryCCDiceCELoss()(torch.zeros(1, 2, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2))
    with pytest.raises(ValueError, match="logits"):
        cc.BinaryCCDiceCELoss()(torch.zeros(1, 3, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2))
