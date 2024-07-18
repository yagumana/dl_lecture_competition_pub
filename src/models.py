import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
# from .utils import wavelet_transform
import sys
# from transformers import Wav2Vec2Model, Wav2Vec2Processor, Trainer, TrainingArguments


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim1: int = 256,
        hid_dim2: int = 64,
        hid_dim3: int = 16
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim1),
            ConvBlock(hid_dim1//4, hid_dim2),
            ConvBlock(hid_dim2//4, hid_dim3)
        )
        
        self.seq_len = seq_len

        self.head = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            Rearrange("b d t -> b (d t)"), #形状を変換
            nn.Linear(hid_dim3*self.seq_len, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, num_classes),
        )

        self.head2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            # Rearrange("b d t -> b (d t)"), #形状を変換
            nn.Linear(hid_dim2, hid_dim1),
            nn.ReLU(),
            # nn.Linear(hid_dim1, hid_dim3),
            # nn.ReLU(),
            nn.Linear(hid_dim1, num_classes),
        )
        # self.head2 = nn.Sequential(
        #     # ここで特徴量を抽出
        #     nn.Linear(hid_dim2 * 6, hid_dim1),  # 統計量が6つあるため
        #     nn.ReLU(),
        #     nn.Linear(hid_dim1, num_classes),
        # )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        # X = self.blocks(X)
        # X = X.permute(0, 2, 1)
        # X, _ = self.rnn(X)
        # X = X.reshape(X.size(0), -1)
        # # print(X.shape)
        # return self.head(X)
        print(X.shape)
        sys.exit()

        b, c, t = X.shape

        self.seq_len = t

        # X = wavelet_transform(X, 'db1', 4)
        X = X.reshape(b, c, -1)

        X = self.blocks(X)  # (b, hid_dim2, t)

        # # 統計量の計算
        # mean = X.mean(dim=-1)  # (b, hid_dim2)
        # std = X.std(dim=-1)  # (b, hid_dim2)
        # max_val = X.max(dim=-1).values  # (b, hid_dim2)
        # min_val = X.min(dim=-1).values  # (b, hid_dim2)
        # range_val = max_val - min_val  # (b, hid_dim2)
        # median = X.median(dim=-1).values  # (b, hid_dim2)

        # # 統計量を結合
        # stats = torch.cat((mean, std, max_val, min_val, range_val, median), dim=-1)  # (b, hid_dim2 * 6)

        return self.head(X)  # (b, num_classes)



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        pool_size: int = 2,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, in_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(in_dim//pool_size, in_dim//pool_size, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=in_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=in_dim//2)

        self.dropout = nn.Dropout(p_drop)

        # プーリング層を追加
        self.maxpool = nn.MaxPool1d(pool_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # if self.in_dim == self.out_dim:
        #     X = self.conv0(X) + X  # skip connection
        # else:
        #     X = self.conv0(X)
        X = self.conv0(X)
        X = self.conv0(X) + X # skip connection

        X = self.dropout(F.gelu(self.batchnorm0(X)))
        X = self.maxpool(X)  # プーリング層を適用

        X = self.conv1(X) + X  # skip connection
        X = self.dropout(F.gelu(self.batchnorm1(X)))
        X = self.maxpool(X)  # プーリング層を適用

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return X
    

# class EEGClassifier(nn.Module):
#     def __init__(self, num_classes: int, feature_dim: int, time_seq: int) -> None:
#         super().__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#         self.classifier = nn.Sequential(
#             nn.Linear(self.wav2vec2.config.hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         batch_size, feature_dim, time_seq = X.size()
#         X = X.view(batch_size, -1)  # 形状を (batch_size, feature_dim * time_seq) に変換
#         features = self.wav2vec2(X).last_hidden_state
#         features = features.mean(dim=1)  # 時間次元に沿って平均
#         logits = self.classifier(features)
#         return logits


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvBlockの定義
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), p_drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p_drop)
        self.pool = nn.MaxPool2d(3, 3)  # MaxPoolingを追加
        self.actv = nn.ELU(alpha=1.0)

        # スキップ接続のための1x1の畳み込み層
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        residual = self.skip_conv(x)  # スキップ接続用の入力
        elu = nn.ELU(alpha=1.0)
        x = self.actv((self.batchnorm1(self.conv1(x))))
        # x = self.dropout(x)
        x = self.actv((self.batchnorm2(self.conv2(x))))
        x += residual  # スキップ接続を適用
        x = self.pool(x)  # プーリング層を適用
        # x = self.dropout(x)
        return x

# 2D畳み込みを行うモデルの例
class BasicConv2DClassifier(nn.Module):
    def __init__(self, num_classes=1600, in_channels=272, time_steps=705, hid_dim1=16, hid_dim2=32, hid_dim3=64, kernel_size=(3, 3), p_drop=0.3):
        super().__init__()
        self.conv_block1 = ConvBlock2D(1, hid_dim1, kernel_size, p_drop)
        self.conv_block2 = ConvBlock2D(hid_dim1, hid_dim2, kernel_size, p_drop)
        self.conv_block3 = ConvBlock2D(hid_dim2, hid_dim3, kernel_size, p_drop)
        
        # 畳み込みとプーリングの後の最終的な形状を計算
        final_dim1 = in_channels // 27  # 2回のプーリングでサイズが1/4になる
        final_dim2 = time_steps // 27   # 2回のプーリングでサイズが1/4になる
        
        # 最終的な全結合層
        print(f"hid_dim2: {hid_dim3}")
        print(f"final_dim1: {final_dim1}")
        print(f"final_dim2: {final_dim2}")
        self.fc1 = nn.Linear(hid_dim3 * final_dim1 * final_dim2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.actv = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(p_drop)



    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, channels, time_steps)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.actv((self.fc1(x)))
        x = self.dropout(x)
        x = self.actv(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

# モデルのインスタンスを作成し、入力データで形状を確認
model = BasicConv2DClassifier()
input_data = torch.randn(16, 272, 705)
output = model(input_data)
print(output.shape)
