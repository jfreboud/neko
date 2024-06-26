import typer
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated

from neko.eval import eval
from neko.dataset import ECGDataset
from neko.model.encoder import ResNet
from neko.model.decoder import Transformer

from config import TEST_FOLDS, MODEL_ARGS, ModelComplexity


def main_eval(
    database_path: Annotated[
        Path, typer.Option("--db", help="Path to the PTB-XL database.")
    ],
    encoder_path: Annotated[
        Path,
        typer.Option(
            "--encoder", help="Path to load the encoder from the disk."
        ),
    ],
    decoder_path: Annotated[
        Path,
        typer.Option(
            "--decoder", help="Path to load the decoder from the disk."
        ),
    ],
    device: Annotated[str, typer.Option(help="Device used to run the models.")],
    model_config: Annotated[
        ModelComplexity, typer.Option("--model", help="Model config to run.")
    ],
):
    """
    Entry point for evaluating the encoder quality.

    Parameters
    ----------
    database_path: Path
        Path to the PTB-XL database (in fact a file).
    encoder_path: Path
        Path to load the encoder from the disk.
    decoder_path: Path
        Path to load the decoder from the disk.
    device: str
        Device used to train the models.
    model_config: ModelComplexity
        Model config to run.
    """
    logger.info(
        f"Main eval launched with following parameters: device={device}, "
        f"model_config={model_config}."
    )
    data = ECGDataset(path=database_path, sampling_rate=100)

    # Split data into train and test.
    test_folds = TEST_FOLDS
    ids_test = np.where(data.df.strat_fold.isin(test_folds))[0]
    data_test = data.filter(ids_test)

    resnet_args = MODEL_ARGS[model_config]["encoder"]
    encoder = ResNet(args=resnet_args)

    transformer_args = MODEL_ARGS[model_config]["decoder"]
    decoder = Transformer(args=transformer_args)

    encoder.load_state_dict(
        torch.load(encoder_path.expanduser(), map_location="cpu")
    )
    decoder.load_state_dict(
        torch.load(decoder_path.expanduser(), map_location="cpu")
    )

    eval(data=data_test, encoder=encoder, decoder=decoder, device=device)
    logger.info("Main eval ended successfully.")


if __name__ == "__main__":
    typer.run(main_eval)
