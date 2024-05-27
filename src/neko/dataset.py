import wfdb
import torch
import pandas
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

from neko.preprocess import Preprocess


class ECGDataset(Dataset):
    """
    A dataset of ECGs.

    Parameters
    ----------
    path: Path
        Path to the data on the disk.
    sampling_rate: int
        The sampling rate of the ECGs.
    df: pandas.DataFrame
        Dataframe used to create a dataset instead of
        going through the `path` normal API.
    """

    def __init__(
        self,
        path: Path,
        sampling_rate: int,
        df: Optional[pandas.DataFrame] = None,
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
            self.df = pd.read_csv(path, index_col="ecg_id")

    def filter(self, indices: np.ndarray) -> "ECGDataset":
        """
        Filter out some ECGs of the current dataset and create a new one.

        Parameters
        ----------
        indices: np.ndarray
            Array of indices to keep in the current dataset.

        Returns
        -------
        _: ECGDataset
            The created new dataset with `indices` lines.
        """
        df = self.df.iloc[indices]
        return ECGDataset(
            path=self.path, sampling_rate=self.sampling_rate, df=df
        )

    def __len__(self):
        """
        Number of elements in the dataset.
        """
        return len(self.df)

    def __getitem__(self, item: int) -> torch.Tensor:
        """
        Get some sample in the dataset.

        Parameters
        ----------
        item: int
            The index of the element to get.

        Returns
        -------
        X: torch.Tensor
            The preprocessed ECG tensor of shape (1, nb_lead)
            where nb_lead = 12.
        """
        if self.sampling_rate == 100:
            data, _ = wfdb.rdsamp(
                (self.dir / self.df.filename_lr.iloc[item]).as_posix()
            )
        else:
            data, _ = wfdb.rdsamp(
                (self.dir / self.df.filename_hr.iloc[item]).as_posix()
            )

        X = torch.Tensor(data)
        X = self.preprocess(X)
        return X
