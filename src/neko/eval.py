import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from neko.generate import generate
from neko.dataset import ECGDataset


def eval(
    data: ECGDataset,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: str,
    chunks: int = 10,
    batch_size: int = 32,
):
    """
    Evaluate an encoder quality.
    The goal is to compare:
        original ECG
            to
        ECG -> encoder -> decoder.

    Parameters
    ----------
    data: ECGDataset
        Dataset of ECGs.
    encoder: Path
        Encoder model to test.
    decoder: Path
        Decoder model to run.
    device: str
        Device used to train the models.
    chunks: int
        Number of split of each ECG in order to limit memory consumption.
    batch_size: int
        Batch size for training.
    """
    logger.info(
        f"Begin eval with following parameters: device={device}, "
        f"chunks={chunks}, batch_size={batch_size}."
    )
    plots_dir = Path(__file__).parent.parent.parent / "data" / "out" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        multiprocessing_context=None,
    )

    encoder.eval()
    decoder.eval()
    encoder.to(device)
    decoder.to(device)

    criterion = torch.nn.MSELoss()

    # At first this function iteration was meant to iterate through the
    # whole test set.
    # But considering the nature of the generation, I only looked qualitatively
    # at some examples and kept this code.
    with torch.no_grad():
        for X in tqdm(
            dataloader,
            unit="batch",
            colour="green",
            mininterval=10,
            maxinterval=60,
        ):
            X = X.to(device)
            y_truth = X.clone().detach()

            batch_dir = plots_dir / "0"
            batch_dir.mkdir(parents=True, exist_ok=True)

            # Reproduce the split that was done during training.
            seq = X.shape[1] // chunks
            for chunk in range(chunks):
                X1 = X[:, chunk * seq : (chunk + 1) * seq, :]
                y1_truth = y_truth[:, chunk * seq : (chunk + 1) * seq, :]
                zero = torch.zeros(
                    (X1.shape[0], 1, X1.shape[2]), device=X1.device
                )

                h1 = encoder(X1)  # compute the style vector
                # h1 = torch.zeros_like(h1)  # uncomment for sanity check
                # h1[:, 0] = 0.001  # uncomment for sanity check

                # Generate curve from scratch (given context is just 0),
                # only taking into account the style vector.
                y1 = generate(
                    nb_points=y1_truth.shape[1],
                    curve=zero,  # X1[:, :10, :]
                    h=h1,
                    decoder=decoder,
                )

                # Save few examples of generated curves.
                val = y1.detach().cpu().numpy()
                for patient in range(5):
                    for lead in range(4):
                        plt.figure()
                        plt.plot(val[patient, :, lead])
                        plt.savefig(batch_dir / f"pat{patient}_lead{lead}.png")
                        plt.close()

                # Save few examples of ground truth.
                val = y1_truth.cpu().numpy()
                for patient in range(5):
                    for lead in range(4):
                        plt.figure()
                        plt.plot(val[patient, :, lead])
                        plt.savefig(
                            batch_dir / f"pat{patient}_lead{lead}_truth.png"
                        )
                        plt.close()

                loss = criterion(y1, y1_truth)
                loss_value = loss.item()
                logger.info(f"Loss: {loss_value}")
                break
            break

    logger.info("Eval ended successfully.")
