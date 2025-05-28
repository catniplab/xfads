import torch


def triangular_inverse(A: torch.Tensor, upper: bool = False):
    eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return torch.linalg.solve_triangular(A, eye, upper=upper)


def bmv(A, x):
    return (A @ x[..., None]).squeeze(-1)


def bop(x1, x2):
    return torch.einsum("...i, ...j -> ...ij", x1, x2)


def bip(x1, x2):
    return torch.sum(x1 * x2, dim=-1)


def bqp(A, x):
    return torch.einsum("...i, ...ij, ...j -> ...", x, A, x)


def chol_bmv_solve(chol_f, v):
    return torch.cholesky_solve(v.unsqueeze(-1), chol_f).squeeze(-1)
