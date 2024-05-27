import torch


def generate(
    nb_points: int,
    curve: torch.Tensor,
    h: torch.Tensor,
    decoder: torch.nn.Module,
) -> torch.Tensor:
    """
    Generate an ECG out of a style vector.

    Parameters
    ----------
    nb_points: int
        Number of points of the ECG to generate.
    curve: int
        Curve given as a context to make the generation
        (like the prompt of LLM).
    h: torch.Tensor
        Hidden features of shape (B, hidden): the style vector.
    decoder: torch.nn.Module
        The decoder model.

    Returns
    -------
    y: torch.Tensor
        The generated curve of shape (B, nb_points, nb_lead) where
        nb_lead = 12.
    """
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
