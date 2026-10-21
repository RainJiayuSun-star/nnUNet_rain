"""CUDA CC/Voronoi geometry adapted from TIO-IKIM/Learning-to-Look-Closer.

Upstream: e5013bdb560eee544a94e1f068cb5cb4d003a51c, Apache-2.0 (see LICENSE).
Changes: lazy dependencies, explicit device/stream, and an empty-mask guard.
No CPU training fallback: CPU geometry belongs only in the test reference.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _cupy_modules():
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import distance_transform_edt, label
    except Exception as exc:
        raise RuntimeError(
            "CC-DiceCE training requires CUDA-enabled CuPy with "
            "cupyx.scipy.ndimage.label and distance_transform_edt. Install the "
            "CuPy wheel matching your CUDA installation (see the CC proposal). "
            "Ordinary nnU-Net training and prediction do not require CuPy."
        ) from exc
    return cp, distance_transform_edt, label


def require_cc_backend(device: torch.device) -> dict:
    """Fail before model allocation if the selected training backend is unusable."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CC-DiceCE training requires an NVIDIA CUDA device; CPU/MPS are unsupported.")
    cp, _, _ = _cupy_modules()
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        with cp.cuda.Device(index):
            cp.zeros(1)  # Check the runtime/driver, not just the Python import.
            runtime = cp.cuda.runtime.runtimeGetVersion()
    except Exception as exc:
        raise RuntimeError("CuPy cannot allocate on the selected CUDA device. Check its CUDA/driver compatibility.") from exc
    return {"cupy": cp.__version__, "cupy_cuda_runtime": runtime}


@torch.no_grad()
def compute_voronoi(mask: torch.Tensor) -> torch.Tensor:
    """Return int64 nearest-GT-component IDs for one [D,H,W] CUDA mask.

    Uses 26-connectivity and Euclidean voxel distances, not physical spacing.
    All-zero masks return zeros without an EDT. Ties use CuPy's tie-breaking.
    """
    if mask.ndim != 3 or mask.device.type != "cuda":
        raise ValueError("CC geometry expects one [D,H,W] CUDA mask.")
    cp, distance_transform_edt, label = _cupy_modules()
    index = mask.device.index
    # DLPack conversion and all CuPy work share PyTorch's active stream.
    with cp.cuda.Device(index), cp.cuda.ExternalStream(torch.cuda.current_stream(mask.device).cuda_stream):
        binary = cp.from_dlpack(mask.contiguous()) > 0
        components, count = label(binary, structure=cp.ones((3, 3, 3), dtype=bool))
        if int(count) == 0:
            return torch.from_dlpack(cp.zeros(binary.shape, dtype=cp.int64))
        indices = distance_transform_edt(
            components == 0, return_distances=False, return_indices=True,
            float64_distances=False,
        )
        return torch.from_dlpack(components[tuple(indices)].astype(cp.int64, copy=False))
