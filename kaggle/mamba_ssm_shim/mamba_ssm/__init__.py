"""Pure-PyTorch stand-in for the `mamba_ssm` package (used only if the CUDA build fails).

It provides `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` with the same maths as the
official CUDA kernel (sequential selective scan), so MambaVision runs on any GPU, more slowly.
If you use it, say so in the paper (training speed differs; results are mathematically the same).
"""
__version__ = "0.0-shim"
