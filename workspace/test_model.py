import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """Transformer模型

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, vocab_size, d_model=512, n_head=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state=None):
        # 输入形状为 (batch_size, sequence_length)
        X = self.emb(inputs)  # 输出形状为 (batch_size, sequence_length, d_model)
        X = X.permute(1, 0, 2)  # 转换为 (sequence_length, batch_size, d_model)
        for layer in self.transformer_layers:
            X = layer(X)
        X = X.permute(1, 0, 2)  # 转换回 (batch_size, sequence_length, d_model)
        output = self.linear(X.reshape((-1, X.shape[-1])))  # 输出形状为 (batch_size, sequence_length, vocab_size)
        return output, None

    def begin_state(self, device, batch_size=1):
        # Transformer 不需要显式的初始状态
        return None