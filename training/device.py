from __future__ import annotations

import torch


def resolve_torch_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build.")
        if device.type == "mps" and not mps_is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def configure_torch_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
