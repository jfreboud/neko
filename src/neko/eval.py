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

            seq = X.shape[1] // chunks
            for chunk in range(chunks):
                X1 = X[:, chunk * seq : (chunk + 1) * seq, :]
                y1_truth = y_truth[:, chunk * seq : (chunk + 1) * seq, :]

                h1 = encoder(X1)
                y1 = generate(
                    nb_points=y1_truth.shape[1],
                    curve=X1[:, :1, :],
                    h=h1,
                    decoder=decoder,
                )

                batch_dir = plots_dir / f"{chunk}"
                batch_dir.mkdir(parents=True, exist_ok=True)

                val = y1.detach().cpu().numpy()
                for patient in range(3):
                    for lead in range(4):
                        plt.figure()
                        plt.plot(val[patient, :, lead])
                        plt.savefig(batch_dir / f"pat{patient}_lead{lead}.png")
                        plt.close()

                val = y1_truth.cpu().numpy()
                for patient in range(3):
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

    logger.info("Eval ended successfully.")
