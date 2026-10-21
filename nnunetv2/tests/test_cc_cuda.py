"""CUDA acceptance checks. Missing CuPy on a CUDA host is a failure, not a skip."""

import os
from time import perf_counter

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.ndimage import distance_transform_edt, label
from torch.nn.parallel import DistributedDataParallel

from nnunetv2.tests.test_cc_losses import reference_voronoi
from nnunetv2.training.loss import instance_losses as cc
from nnunetv2.utilities import connected_components as geometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance requires an NVIDIA GPU")


@pytest.mark.parametrize("kind", ["empty", "full", "diagonal", "separate", "ties"])
def test_cuda_geometry(kind):
    geometry.require_cc_backend(torch.device("cuda", 0))
    mask = torch.zeros(8, 8, 8, device="cuda", dtype=torch.int16)
    if kind == "full":
        mask.fill_(1)
    elif kind == "diagonal":
        mask[1, 1, 1] = mask[2, 2, 2] = 1
    elif kind in ("separate", "ties"):
        mask[1, 1, 1] = 1
        mask[5 if kind == "ties" else 6, 1, 1] = 1
    # Non-default stream exercises DLPack synchronization and stream ownership.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = geometry.compute_voronoi(mask)
    torch.cuda.current_stream().wait_stream(stream)
    actual = actual.cpu().numpy()
    binary = mask.cpu().numpy() > 0
    components, count = label(binary, structure=np.ones((3, 3, 3)))
    if not count:
        assert not actual.any()
        return
    assert set(np.unique(actual)) == set(range(1, count + 1))
    # Align IDs by their seeds: library component numbering need not match.
    aligned = np.zeros_like(actual)
    for k in range(1, count + 1):
        ids = np.unique(actual[components == k])
        assert len(ids) == 1
        aligned[actual == ids[0]] = k
    distances = np.stack([distance_transform_edt(components != k) for k in range(1, count + 1)])
    assigned = np.take_along_axis(distances, (aligned - 1)[None], axis=0)[0]
    np.testing.assert_allclose(assigned, distances.min(0))
    if kind != "ties":
        np.testing.assert_array_equal(aligned, reference_voronoi(mask).cpu().numpy())


def test_empty_geometry_does_not_call_edt(monkeypatch):
    cp, _, label_fn = geometry._cupy_modules()

    def forbidden(*args, **kwargs):
        raise AssertionError("EDT must not run for an empty mask")

    monkeypatch.setattr(geometry, "_cupy_modules", lambda: (cp, forbidden, label_fn))
    assert not geometry.compute_voronoi(torch.zeros(4, 4, 4, device="cuda")).any()


@pytest.mark.parametrize("include_global", [False, True])
@pytest.mark.parametrize("compile_network", [False, True])
def test_cuda_forward_backward_amp(include_global, compile_network):
    geometry.require_cc_backend(torch.device("cuda", 0))
    network = torch.nn.Conv3d(2, 3, 1).cuda()
    if compile_network:
        network = torch.compile(network)
    x = torch.randn(2, 2, 8, 8, 8, device="cuda")
    y = torch.zeros(2, 1, 8, 8, 8, device="cuda", dtype=torch.int16)
    y[0, 0, 1, 1, 1] = 1
    y[0, 0, 6, 1, 1] = 1
    y[1, 0, 2:5, 3:5, 3:5] = 2
    loss_fn = cc.TwoLabelCCDiceCELoss(include_global)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        output = network(x)
        loss = loss_fn(output, y)
    # The loss must disable the surrounding AMP context itself.
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    torch.testing.assert_close(loss, loss_fn(output.float(), y))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    scaler.step(optimizer)
    scaler.update()


def _ddp_worker(rank, rendezvous, include_global):
    torch.cuda.set_device(rank)
    geometry.require_cc_backend(torch.device("cuda", rank))
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(8)
        model = torch.nn.Conv3d(2, 3, 1).cuda(rank)
        reference = torch.nn.Conv3d(2, 3, 1).cuda(rank)
        reference.load_state_dict(model.state_dict())
        network = DistributedDataParallel(model, device_ids=[rank])
        x = torch.linspace(-1, 1, 2 * 2 * 8**3, device=rank).reshape(2, 2, 8, 8, 8)
        y = torch.zeros(2, 1, 8, 8, 8, device=rank, dtype=torch.int16)
        y[0, 0, 1, 1, 1] = 1
        y[0, 0, 6, 1, 1] = 1
        y[1, 0, 2:5, 3:5, 3:5] = 2
        # Equal samples per rank, different component counts and absent classes.
        loss = cc.TwoLabelCCDiceCELoss(include_global, batch_dice=True, ddp=True)
        loss(network(x[rank:rank+1]), y[rank:rank+1]).backward()
        expected = cc.TwoLabelCCDiceCELoss(include_global, batch_dice=True, ddp=False)
        expected(reference(x), y).backward()
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-5, rtol=2e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("include_global", [False, True])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DDP acceptance requires two CUDA GPUs")
def test_cuda_ddp(tmp_path, include_global):
    rendezvous = (tmp_path / "ddp_init").as_uri()
    mp.spawn(_ddp_worker, args=(rendezvous, include_global), nprocs=2, join=True)


def test_cuda_loss_profile():
    """Report loss-only overhead, not a complete training-step memory estimate."""
    geometry.require_cc_backend(torch.device("cuda", 0))
    shape = tuple(int(v) for v in os.environ.get("CC_BENCH_SHAPE", "16,16,16").split(","))
    assert len(shape) == 3 and min(shape) >= 8
    y = torch.zeros(1, 1, *shape, device="cuda", dtype=torch.int16)
    y[0, 0, 1, 1, 1] = 1
    y[0, 0, -2, -2, -2] = 1
    y[0, 0, 3:6, 3:6, 3:6] = 2
    logits = torch.randn(1, 3, *shape, device="cuda", requires_grad=True)
    for include_global in (False, True):
        loss = cc.TwoLabelCCDiceCELoss(include_global)
        loss(logits, y).backward()  # Warm up CuPy kernels.
        logits.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(3):
            logits.grad = None
            loss(logits, y).backward()
        torch.cuda.synchronize()
        cp, _, _ = geometry._cupy_modules()
        print({"global": include_global, "shape": shape,
               "seconds_per_loss_backward": (perf_counter() - start) / 3,
               "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "cupy_pool_reserved_bytes": cp.get_default_memory_pool().total_bytes()})
