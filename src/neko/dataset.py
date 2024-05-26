import wfdb
import torch
import pandas
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset

from neko.preprocess import Preprocess


class ECGDataset(Dataset):
    """
    A dataset of labeled images.

    Parameters
    ----------
    directories: [str]
        List of directories which correspond to actual slide with annotations.
    label: int
        The label to select inside the different slide directories.
    transform
        The transform to operate on each image.
    """

    def __init__(
        self,
        path: Path,
        sampling_rate: int,
        df: Optional[pandas.DataFrame] = None
    ):
        self.path = path.expanduser()
        self.dir = self.path.parent
        self.preprocess = Preprocess(
            True, True, True, should_pre_transpose=True
        )
        self.sampling_rate = sampling_rate

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(path, index_col='ecg_id')

    def filter(self, indices: np.ndarray) -> 'ECGDataset':
        df = self.df.iloc[indices]
        return ECGDataset(
            path=self.path,
            sampling_rate=self.sampling_rate,
            df=df
        )

    def __len__(self):
        """
        Number of elements in the dataset.
        """
        return len(self.df)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        """
        Get some element in the dataset.

        Parameters
        ----------
        item: int
            The index of the element to get.

        Returns
        -------
        A tensor containing the image loaded and transformed.
        """
        if self.sampling_rate == 100:
            data, _ = wfdb.rdsamp((self.dir / self.df.filename_lr.iloc[item]).as_posix())
        else:
            data, _ = wfdb.rdsamp((self.dir / self.df.filename_hr.iloc[item]).as_posix())

        X = torch.Tensor(data)
        X = self.preprocess(X)
        return X
