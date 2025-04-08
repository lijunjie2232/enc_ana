import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "S_Mamba"
data_dir = "./dataset/tst/S-Mamba_datasets/exchange_rate/"
csv_name = "exchange_rate.csv"
# data_type = "custom"
# data_dir = "./dataset/bitcoin-historical-data"
# csv_name = "btcusd_1-min_data.csv"
data_type = "custom"
d_model = 128
expand = 2
d_conv = 4
head_dim = 32
pred_len = 720
seq_len = 720
label_len = 128


subprocess.run(
    [
        "python",
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--root_path",
        data_dir,
        "--data_path",
        csv_name,
        "--model_id",
        "Exchange_96_96",
        "--model",
        model_name,
        "--data",
        data_type,
        "--features",
        "M",
        "--seq_len",
        f"{seq_len}",
        "--pred_len",
        f"{pred_len}",
        "--label_len",
        f"{label_len}",
        "--e_layers",
        "2",
        "--enc_in",
        "8",
        "--dec_in",
        "8",
        "--c_out",
        "8",
        "--des",
        "Exp",
        "--batch_size",
        "3872",
        "--learning_rate",
        "0.005",
        "--d_ff",
        "128",
        "--itr",
        "1",
        "--d_model",
        f"{d_model}",
        "--expand",
        f"{expand}",
        "--d_conv",
        f"{d_conv}",
        "--head_dim",
        f"{head_dim}",
    ],
    check=False,
)
