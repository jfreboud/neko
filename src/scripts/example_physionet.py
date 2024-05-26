import ast
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + "/" + df.filename_lr.iloc[0])]
        data += [wfdb.rdsamp(path + "/" + df.filename_lr.iloc[1])]
        data += [wfdb.rdsamp(path + "/" + df.filename_lr.iloc[2])]
        # data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


if __name__ == "__main__":
    plots_dir = Path(__file__).parent.parent.parent / "data" / "out" / "example"
    plots_dir.mkdir(parents=True, exist_ok=True)

    path = (
        Path(
            "~/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
        )
        .expanduser()
        .as_posix()
    )
    sampling_rate = 100

    # load and convert annotation data
    Y = pd.read_csv(path + "/ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    val = X
    for patient in range(3):
        for lead in range(12):
            plt.figure()
            plt.plot(val[patient, :, lead])
            plt.savefig(plots_dir / f"pat{patient}_lead{lead}.png")
            plt.close()
