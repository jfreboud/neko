from enum import Enum
from neko.model.encoder import ResNetArgs
from neko.model.decoder import TransformerArgs


TEST_FOLDS = [7, 8, 9, 10]


class ModelComplexity(str, Enum):
    SMALL = "small"
    LARGE = "large"
    XLARGE = "xlarge"


RESNET_SMALL = ResNetArgs(
    planes=[16, 32, 64, 128],
    blocks=[3, 4, 6, 3],
)
RESNET_LARGE = ResNetArgs(
    planes=[64, 128, 256, 512],
    blocks=[3, 4, 6, 3],
)

TRANSFORMER_SMALL = TransformerArgs(
    dim=512,
    n_layers=8,
    head_dim=128,
    hidden_dim=9092,
    n_heads=4,
    n_leads=12,
    embedding_dim=512,
)
TRANSFORMER_LARGE = TransformerArgs(
    dim=1024,
    n_layers=8,
    head_dim=128,
    hidden_dim=9092,
    n_heads=8,
    n_leads=12,
    embedding_dim=2048,
)
TRANSFORMER_XLARGE = TransformerArgs(
    dim=1024,
    n_layers=16,
    head_dim=128,
    hidden_dim=9092,
    n_heads=8,
    n_leads=12,
    embedding_dim=2048,
)

RESNET_ARGS = {
    ModelComplexity.SMALL: RESNET_SMALL,
    ModelComplexity.LARGE: RESNET_LARGE,
}

TRANSFORMER_ARGS = {
    ModelComplexity.SMALL: TRANSFORMER_SMALL,
    ModelComplexity.LARGE: TRANSFORMER_LARGE,
    ModelComplexity.XLARGE: TRANSFORMER_XLARGE,
}

MODEL_ARGS = {
    ModelComplexity.SMALL: {
        "encoder": RESNET_ARGS[ModelComplexity.SMALL],
        "decoder": TRANSFORMER_ARGS[ModelComplexity.SMALL],
    },
    ModelComplexity.LARGE: {
        "encoder": RESNET_ARGS[ModelComplexity.LARGE],
        "decoder": TRANSFORMER_ARGS[ModelComplexity.LARGE],
    },
    ModelComplexity.XLARGE: {
        "encoder": RESNET_ARGS[ModelComplexity.LARGE],
        "decoder": TRANSFORMER_ARGS[ModelComplexity.XLARGE],
    },
}
