import torch
from tqdm import tqdm
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
    dataloader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        multiprocessing_context=None,
    )

    encoder.train()
    decoder.train()
    encoder.to(device)
    decoder.to(device)

    criterion = torch.nn.MSELoss()
    optimizer1 = torch.optim.AdamW(
        encoder.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )
    optimizer2 = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

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
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            X = X.to(device)
            y = X.clone().detach()

            seq = X.shape[1] // chunks
            for chunk in range(chunks):
                X1 = X[:, chunk * seq : (chunk + 1) * seq, :]
                y1 = y[:, chunk * seq : (chunk + 1) * seq, :]
                y1 = y1[:, 1:, :]
                X1 = X1[:, :-1, :]

                x1 = encoder(X1)
                x1, _ = decoder(X1, x1)

                loss = criterion(x1, y1) / chunks
                loss_value = loss.item()
                epoch_loss += loss_value
                cur_loss += loss_value
                loss.backward()

            optimizer1.step()
            optimizer2.step()
            nb_steps += 1

            if nb_steps % 10 == 0:
                cur_loss = cur_loss / 10
                logger.info(f"Loss: {cur_loss}.")
                cur_loss = 0

        running_loss = epoch_loss / nb_steps
        logger.info(f"Epoch {epoch+1}/{nb_epochs}, loss: {running_loss}.")
