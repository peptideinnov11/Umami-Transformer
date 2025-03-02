import math
import torch.nn.functional as F  # 导入 F
import torch
import torch.nn as nn
import torch.optim as optim

# 步骤2: 构建模型
class MLPModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=50, dropout_prob=0.5):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
class PositionalEncoding(nn.Module):
    """
    计算输入序列的位置编码。

    Args:
        d_model (int): 词嵌入的维度。
        max_len (int, optional): 可处理的最大序列长度。默认为 5000.
        dropout (float, optional): dropout 概率。默认为 0.1.

    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建包含位置编码的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 使用sin和cos函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将pe转换为合适的维度 (max_len, 1, d_model) 并注册为buffer
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码添加到输入序列。

        Args:
            x (torch.Tensor): 形状为 (batch_size, seq_len, d_model) 的输入序列。

        Returns:
            torch.Tensor: 形状为 (batch_size, seq_len, d_model) 的编码后的序列。
        """
        # 将预先计算好的位置编码添加到输入中
        # pe的形状为(max_len, 1, d_model)，通过广播机制适应不同batch size的输入
        x = x + self.pe[:x.size(1), :].view(1, 50, 8)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclasses, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, max_len=1000, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, nclasses)
        self.fc1 = nn.Linear(ntoken*ninp, 256)
        #self.fc = nn.Linear(ntoken * ninp, 1)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(ntoken * ninp+16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.MLP = MLPModel(input_dim=8, hidden_dim=128, output_dim=16, dropout_prob=0.5)
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cpu')
    def forward(self, src):
        seq_data = src[0].to(self.device)

        seq_data = self.embedding(seq_data) * math.sqrt(self.ninp)
        seq_data = self.pos_encoder(seq_data)
        output = self.transformer_encoder(seq_data)
        # 使用最后一个时间步的输出作为序列表示
        output = output.view(output.size(0), -1)
        #output = self.fc(output)
        output1 = self.MLP(src[1].to(self.device))
        oc = torch.cat((output, output1), dim=1)
        oc = self.fc3(oc)
        # 使用 sigmoid 函数进行二分类
        output = torch.sigmoid(oc)
        return output
