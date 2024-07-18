import random
import numpy as np
import torch
from tqdm import tqdm
import pywt

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import torch
import pywt
import numpy as np


# Wavelet変換の適用
def apply_wavelet_transform(data, wavelet, level):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

# 各Wavelet係数をパディングして同じ形状にする関数
def pad_wavelet_coeffs(coeffs, max_len):
    padded_coeffs = [np.pad(c, (0, max_len - len(c)), 'constant') for c in coeffs]
    return np.stack(padded_coeffs)

def wavelet_transform(X, wavelet, level):
    b, c, t = X.shape
    device = X.device  # 元のデバイスを保持
    X = X.cpu()  # 一時的にCPUに移動

    batch_size, f_dim, t_seq = X.shape
    X_wavelet = []
    max_len = 0
    for i in tqdm(range(batch_size)):
        channel_coeffs = []
        for j in range(f_dim):
            coeffs = apply_wavelet_transform(X[i, j].numpy(), wavelet, level)
            channel_coeffs.append(coeffs)
            # 各スケールの係数の最大長を計算
            max_len = max(max_len, *map(len, coeffs))
        X_wavelet.append(channel_coeffs)

    # パディングされたWavelet係数を保存
    X_wavelet_padded = []
    for batch in tqdm(X_wavelet):
        batch_coeffs = []
        for channel in batch:
            padded_coeffs = pad_wavelet_coeffs(channel, max_len)
            batch_coeffs.append(padded_coeffs)
        X_wavelet_padded.append(batch_coeffs)

    # リストからNumPy配列に変換してからテンソルに変換
    X_wavelet_padded = np.array(X_wavelet_padded)
    # 結果をテンソルに変換
    X_wavelet_padded = torch.tensor(X_wavelet_padded).to(device)
    X_wavelet_padded = X_wavelet_padded.reshape(b, c, -1)

    return X_wavelet_padded




def apply_wavelet_transform(data, scales, wavelet):
    coeffs, freq = pywt.cwt(data, scales, wavelet)
    return coeffs

def cmor_transform(X):
    batch_size, feature_dim, time_seq = X.shape
    # device = X.device  # 元のデバイスを保持
    # X = X.cpu()  # 一時的にCPUに移動

    wavelet = 'cmor'
    scales = np.arange(1, 50)

    X_wavelet = []

    for i in tqdm(range(batch_size)):
        channel_coeffs = []
        for j in range(feature_dim):
            coeffs = apply_wavelet_transform(X[i, j].numpy(), scales, wavelet)
            channel_coeffs.append(coeffs)
           
        channel_coeffs = np.stack(channel_coeffs)
        X_wavelet.append(channel_coeffs)
    X_wavelet = np.stack(X_wavelet)

    X_wavelet = torch.tensor(X_wavelet)

    return X_wavelet

#
# データの読み込み（例としてランダムデータを使用）
# batch_size = 32
# feature_dim = 271
# time_seq = 231
# X = torch.randn(batch_size, feature_dim, time_seq)
# X_wavelet = cmor_transform(X)

# # 変換後のデータの形状を確認
# print("Original shape:", X.shape)
# print("Wavelet transformed shape:", X_wavelet.shape)
