import torch


def euclidean_distance(s: torch.Tensor(float), r: torch.Tensor(float)) -> torch.Tensor(float):
    # s -> source point
    # d -> destination (receiving point)
    s = torch.as_tensor(s)
    r = torch.as_tensor(r)
    return torch.linalg.norm(s - r)

def sum_rule(g: torch.Tensor(complex)) -> torch.Tensor(float):
    # g -> matrix of complex numbers, shape (Nr, Ns)
    # returns S -> float, sum of the absolute squares of the elements of g
    return torch.sum(torch.abs(g)**2)
