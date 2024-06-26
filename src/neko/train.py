import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader

from neko.dataset import ECGDataset


def train(
    data: ECGDataset,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: str,
    chunks: int = 10,
    batch_size: int = 32,
    nb_epochs: int = 4,
):
    """
    Train an encoder and decoder end to end in a generative way.

    Parameters
    ----------
    data: ECGDataset
        Dataset of ECGs.
    encoder: Path
        Encoder model to train.
    decoder: Path
        Decoder model to train.
    device: str
        Device used to train the models.
    chunks: int
        Number of split of each ECG in order to limit memory consumption.
    batch_size: int
        Batch size for training.
    nb_epochs: int
        Number of epochs for training.
    """
    logger.info(
        f"Begin training with following parameters: device={device}, "
        f"chunks={chunks}, batch_size={batch_size}, nb_epochs={nb_epochs}."
    )
    checkpoints_dir = (
        Path(__file__).parent.parent.parent / "data" / "out" / "checkpoints"
    )
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        multiprocessing_context=None,
    )

    # checkpoint_path = checkpoints_dir / "epoch8.pth"
    # checkpoints = torch.load(checkpoint_path, map_location="cpu")
    # encoder.load_state_dict(checkpoints["encoder_state_dict"])
    # decoder.load_state_dict(checkpoints["decoder_state_dict"])

    encoder.train()
    decoder.train()
    encoder.to(device)
    decoder.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3,
        weight_decay=1e-3,
    )
    # optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        cur_loss = 0.0
        nb_steps = 0

        for X in tqdm(
            dataloader,
            unit="batch",
            colour="green",
            mininterval=10,
            maxinterval=60,
        ):
            optimizer.zero_grad()

            X = X.to(device)
            y = X.clone().detach()

            # Split the original signal into chunks in order
            # to lower memory consumption (see reasoning in the README.md).
            seq = X.shape[1] // chunks
            for chunk in range(chunks):
                X1 = X[:, chunk * seq : (chunk + 1) * seq, :]
                y1 = y[:, chunk * seq : (chunk + 1) * seq, :]

                # The ground truth is the input signal but in the "future".
                y1 = y1[:, 1:, :]  # the ground truth
                X1 = X1[:, :-1, :]  # the input signal

                x1 = encoder(X1)  # compute the style vector
                x1, _ = decoder(X1, x1)  # generate new points

                # Compare new points to ground truth.
                # Divide by number of chunks because the gradient
                # is accumulated.
                loss = criterion(x1, y1) / chunks
                loss_value = loss.item()

                epoch_loss += loss_value
                cur_loss += loss_value

                # Accumulate gradient (do not call zero_grad).
                loss.backward()

            # Update parameters.
            optimizer.step()
            nb_steps += 1

            if nb_steps % 10 == 0:
                cur_loss = cur_loss / 10
                logger.info(f"Loss: {cur_loss}.")
                cur_loss = 0

        running_loss = epoch_loss / nb_steps
        logger.info(f"Epoch {epoch + 1}/{nb_epochs}, loss: {running_loss}.")

        # checkpoint_path = checkpoints_dir / f"epoch{epoch + 1 + 8}.pth"
        # torch.save(
        #     {
        #         "epoch": epoch + 1 + 8,
        #         "encoder_state_dict": encoder.state_dict(),
        #         "decoder_state_dict": decoder.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "loss": {running_loss},
        #     },
        #     checkpoint_path.as_posix(),
        # )

    logger.info("Training ended successfully.")
