import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
from tqdm import tqdm

SYMBOLS = {
    i: idx
    for idx, i in enumerate(
        [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "WIFUSDT",
            "DOGEUSDT",
            "PEPEUSDT",
            "SPELLUSDT",
            "SUIUSDT",
            "ADAUSDT",
            "RVNUSDT",
            "JUVUSDT",
            "OMUSDT",
            "LTCUSDT",
            "CREAMUSDT",
            "ACMUSDT",
            "CITYUSDT",
            "TSTUSDT",
            "ATMUSDT",
            "USUALUSDT",
            "DATAUSDT",
            "PORTOUSDT",
            "BARUSDT",
            "TRXUSDT",
            "XLMUSDT",
            "LINKUSDT",
            "JUPUSDT",
            "BNXUSDT",
            "PNUTUSDT",
            "CAKEUSDT",
            "SHIBUSDT",
            "WBTCUSDT",
            "AVAXUSDT",
            "HBARUSDT",
            "TONUSDT",
            "DOTUSDT",
        ]
    )
}

# 读取和预处理 CSV 数据
def load_and_preprocess_data(file_path, interval):
    # 读取 CSV 文件
    data = pd.read_csv(file_path).dropna()

    # 处理缺失值
    data.fillna(value=0, inplace=True)  # 使用前向填充方法处理缺失值

    # # 将 datetime 列转换为时间戳
    # data["timestamp"] = (
    #     pd.to_datetime(data["datetime"]).astype("int64") // 10**9
    # )  # 转换为秒级时间戳

    # # 排序 timestamp
    # data = data.sort_values(by="timestamp", ascending=True)

    # # make index
    # # interval = 14400
    # data["timestamp_diff"] = data["timestamp"].diff().fillna(0).astype("int64")

    # use csv in bn_spider
    interval *= 1000
    data = data.sort_values(by="open_time", ascending=True)
    data["timestamp_diff"] = data["open_time"].diff().fillna(0).astype("int64")

    # seq = pd.DataFrame(columns=["start", "end", "length"])
    # start = 0
    # seq_list = []
    # idx = 0
    # row = None
    # for idx, row in tqdm(data.iterrows(), total=len(data)):
    #     if row.timestamp_diff != interval:
    #         if idx == 0:
    #             continue
    #         seq_list.append([start, idx, idx - start])
    #         start = idx
    # if idx > 0:
    #     idx += 1
    #     seq_list.append([start, idx, idx - start])
    # seq = pd.DataFrame(seq_list, columns=["start", "end", "length"])

    # 使用 shift 和 cumsum 来划分序列
    data["is_new_seq"] = (data["timestamp_diff"] != interval).astype(int)
    data["seq_id"] = data["is_new_seq"].cumsum()

    # 计算每个序列的 start, end, length
    seq = (
        data.groupby("seq_id")
        .agg(
            start=("open_time", "idxmin"),
            end=("open_time", "idxmax"),
            length=("open_time", "size"),
        )
        .reset_index(drop=True)
    )
    seq["end"] += 1

    # gen_seq = generate_fixed_length_sequences(seq, 5)

    # # 提取特征
    # features = data[["open", "high", "low", "close", "volume"]].values

    # # 归一化处理
    # scaler = MinMaxScaler()
    # features_scaled = scaler.fit_transform(features)

    # return data[["open", "high", "low", "close", "volume"]].values, seq

    return (
        data[
            [
                "stock_name",
                # "open_time",
                "interval",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                # "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
            ]
        ].values,
        seq,
    )


def generate_fixed_length_sequences(df, fixed_length):
    """
    将每个连续区间分割为固定长度的子序列，返回无交叉的区间列表

    参数:
    df (DataFrame): 包含列['start', 'end', 'length']的连续性统计表
    fixed_length (int): 需要生成的固定子序列长度

    返回:
    List[Tuple]: 生成的子序列区间列表，格式为 [(start1, end1), (start2, end2), ...]
    """
    result = []

    for _, row in df.iterrows():
        s = row["start"]
        e = row["end"]
        length = row["length"]

        # 跳过长度不足的区间
        if length < fixed_length:
            continue

        # 完全匹配的情况
        if length == fixed_length:
            result.append((s, e))
            continue

        # form offset
        if length // fixed_length == 1:
            offset = np.random.randint(0, length % fixed_length + 1)
        else:
            offset = np.random.randint(0, fixed_length)

        # 生成每个子序列的起始和结束位置
        s += offset
        e = s + fixed_length
        while e <= length:
            result.append((s, e))
            s = e
            e += fixed_length

    return result


# 创建自定义数据集类
class SeqDataset(Dataset):
    def __init__(self, data, seq, sequence_length, predict_length):
        self.data = data
        self.seq = seq
        self.seqList = None
        symbol_mapper = np.vectorize(lambda s: SYMBOLS[s])
        self.data[:, 0] = symbol_mapper(self.data[:, 0])
        self.data = self.data.astype(np.float64)
        self.sequence_length = sequence_length
        self.predict_length = predict_length

        self.__shuffle__()

    def __shuffle__(self):
        self.seqList = generate_fixed_length_sequences(
            self.seq, self.sequence_length + self.predict_length
        )
    
    def reshuflle(self):
        self.__shuffle__()

    def __len__(self):
        return len(self.seqList)

    def __getitem__(self, idx):
        seq = self.seqList[idx]
        data = self.data[seq[0] : seq[1]]

        x = data[0 : self.sequence_length]
        y = data[self.sequence_length :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# 定义 TransformerModel
class TransformerModel(pl.LightningModule):
    """Transformer模型

    Defined in :numref:`sec_rnn-concise`"""

    def __init__(
        self,
        input_dim,
        d_model=512,
        n_head=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        lr=0.001,
        predict_length=10,
    ):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()
        self.num_hiddens = d_model
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.symbol_emb = nn.Embedding(len(SYMBOLS), d_model)
        self.interval_emb = nn.Embedding(604800, d_model)
        self.linear = nn.Linear(d_model, input_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        # 输入形状为 (batch_size, sequence_length, input_dim)
        X = inputs.permute(1, 0, 2)  # 转换为 (sequence_length, batch_size, input_dim)
        for layer in self.transformer_layers:
            X = layer(X)
        X = X.permute(1, 0, 2)  # 转换回 (batch_size, sequence_length, input_dim)
        output = self.linear(X)  # 输出形状为 (batch_size, sequence_length, input_dim)
        return output

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = nn.MSELoss()(outputs, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = nn.MSELoss()(outputs, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


# 创建数据模块
class StockDataModule(pl.LightningDataModule):
    def __init__(self, file_path, interval, sequence_length, predict_length, batch_size):
        super().__init__()
        self.file_path = file_path
        self.interval = interval
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.batch_size = batch_size
        self.features_scaled = None
        self.vseq = None
        self.seqList = None

        self.setup()

    def prepare_data(self):
        # 读取和预处理数据
        self.features_scaled, self.vseq = load_and_preprocess_data(self.file_path, self.interval)

    def setup(self, stage=None):
        if self.features_scaled is None or self.vseq is None:
            self.prepare_data()
        # 创建数据集
        self.dataset = SeqDataset(
            self.features_scaled,
            self.vseq,
            self.sequence_length,
            self.predict_length,
        )

    def train_dataloader(self):
        self.dataset.reshuflle()
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    from pathlib import Path

    # 参数设置
    file_path = "/home/ljj/code/py/nlp/mamba/dataset/bn_spider/csv/BTCUSDT_60.csv"  # 替换为你的 CSV 文件路径

    ROOT = Path(__file__).parent.resolve()
    # 读取 CSV 文件
    # data, seq = load_and_preprocess_data(ROOT / "test_data.csv", 14400)

    sequence_length = 60  # 序列长度
    predict_length = 10  # 预测长度

    input_dim = 5  # 输入特征数量
    d_model = 512
    n_head = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    batch_size = 32
    num_epochs = 50
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建数据模块
    data_module = StockDataModule(
        file_path, 60, sequence_length, predict_length, batch_size
    )

    # 初始化模型
    model = TransformerModel(
        input_dim, d_model, n_head, num_layers, dim_feedforward, dropout, lr
    )

    # 创建 Trainer
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=1 if device == "cuda" else 0)

    # 训练模型
    trainer.fit(model, data_module)

    # 保存模型
    torch.save(model.state_dict(), "transformer_model.pth")

    # 预测
    model.eval()
    with torch.no_grad():
        # 假设我们使用最后一个序列进行预测
        last_sequence = (
            torch.tensor(
                data_module.features_scaled[-sequence_length:], dtype=torch.float32
            )
            .unsqueeze(0)
            .to(device)
        )
        predicted = model(last_sequence)
        predicted = predicted.squeeze(0).cpu().numpy()

        # 反归一化
        predicted = data_module.scaler.inverse_transform(predicted)
        print("Predicted sequence:", predicted)
