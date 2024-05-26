import torch


def generate(
    X: torch.Tensor,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module
) -> torch.Tensor:
    y = torch.zeros_like(X)
    y[:, 0, :] = X[:, 0, :]

    h = encoder(X)
    x = y[:, 0, :]
    x = x[:, None, :]

    x, cache = decoder(x, h)
    y[:, 1, :] = x[:, 0, :].detach()

    for i in range(1, X.shape[1]-1):
        x = y[:, i, :]
        x = x[:, None, :]

        x, cache = decoder(x, cache=cache)
        y[:, i+1, :] = x[:, 0, :].detach()

    return y
