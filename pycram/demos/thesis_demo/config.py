import torch


def select_backend():
    """Return "cuda" when PyTorch reports CUDA support, otherwise "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
