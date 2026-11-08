import torch
import torch.nn.functional as F


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """u, delta: (b, d, L); A: (d, n); B, C: (d, n) or (b, n, L) or (b, g, n, L); D: (d,); z: (b, d, L)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, L = u.shape
    A, B, C = A.float(), B.float(), C.float()

    def expand(M):      # (b, g, n, L) -> (b, d, n, L)
        return M.repeat_interleave(d // M.shape[1], dim=1)

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if B.dim() == 2:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, expand(B), u)
    if C.dim() == 4:
        C = expand(C)
    x = u.new_zeros((b, d, A.shape[1]))
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 2:
            ys.append(torch.einsum("bdn,dn->bd", x, C))
        elif C.dim() == 3:
            ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
        else:
            ys.append(torch.einsum("bdn,bdn->bd", x, C[:, :, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.float()[..., None]
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype_in)
    return (out, x) if return_last_state else out


selective_scan_fn = selective_scan_ref
