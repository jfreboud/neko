import torch


def generate(
    nb_points: int,
    curve: torch.Tensor,
    h: torch.Tensor,
    decoder: torch.nn.Module,
) -> torch.Tensor:
    if nb_points < 2:
        raise ValueError("At least 2 points should be generated.")

    y = torch.zeros(
        (h.shape[0], nb_points, curve.shape[2]), device=curve.device
    )
    n_given = curve.shape[1]
    y[:, :n_given, :] = curve

    x = y[:, :n_given, :]
    x, cache = decoder(x, h)
    y[:, n_given, :] = x[:, n_given - 1, :].detach()

    for i in range(n_given, nb_points - n_given - 1):
        x = y[:, i, :][:, None, :]
        x, cache = decoder(x, cache=cache)
        y[:, i + 1, :] = x[:, 0, :].detach()

    return y
