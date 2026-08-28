from __future__ import annotations


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run: python -m pip install -r requirements-torch-cu128.txt"
        ) from exc

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda_version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this environment. Check that you activated the Conda env "
            "and installed the CUDA PyTorch wheel."
        )

    device = torch.device("cuda")
    print(f"device_name: {torch.cuda.get_device_name(0)}")
    x = torch.rand(4, 4, device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(f"gpu_tensor_test_sum: {y.sum().item():.6f}")
    print(f"allocated_mb: {torch.cuda.memory_allocated(0) / 1024**2:.2f}")
    print(f"reserved_mb: {torch.cuda.memory_reserved(0) / 1024**2:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
