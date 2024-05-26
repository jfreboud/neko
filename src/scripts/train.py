import typer
import torch
import numpy as np
from pathlib import Path
from typing_extensions import Annotated

from neko.train import train
from neko.dataset import ECGDataset
from neko.model.encoder import ResNet
from neko.model.decoder import Transformer

from config import TEST_FOLDS, MODEL_ARGS, ModelComplexity


def main_train(
    database_path: Annotated[Path, typer.Option("--db", help="Path to the PTXL database.")],
    encoder_path: Annotated[Path, typer.Option("--encoder", help="Path to save the encoder on the disk.")],
    decoder_path: Annotated[Path, typer.Option("--decoder", help="Path to save the decoder on the disk.")],
    device: Annotated[str, typer.Option(help="Device used to train the models.")],
    model_config: Annotated[ModelComplexity, typer.Option("--model", help="Model config to train.")]
):
    data = ECGDataset(
        path=database_path,
        sampling_rate=100
    )

    # Split data into train and test.
    test_folds = TEST_FOLDS
    ids_train = np.where(~data.df.strat_fold.isin(test_folds))[0]
    data_train = data.filter(ids_train)

    resnet_args = MODEL_ARGS[model_config]["encoder"]
    encoder = ResNet(args=resnet_args)

    transformer_args = MODEL_ARGS[model_config]["decoder"]
    decoder = Transformer(args=transformer_args)

    train(
        data=data_train,
        encoder=encoder,
        decoder=decoder,
        device=device
    )

    torch.save(encoder.state_dict(), encoder_path.expanduser())
    torch.save(decoder.state_dict(), decoder_path.expanduser())


if __name__ == "__main__":
    typer.run(main_train)
