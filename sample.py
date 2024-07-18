import os 
import torch
import mne
from sklearn.preprocessing import StandardScaler

def load_data(data_dir: str, split: str):
    X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
    subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt")) 
    return X, subject_idxs

def check_channel_info(X):
    # データの形状を取得
    n_samples, n_channels, n_times = X.shape

    # MNE を使用して RawArray を作成
    info = mne.create_info(ch_names=[str(i) for i in range(n_channels)], sfreq=200, ch_types='eeg')
    raw = mne.io.RawArray(X.permute(1, 0, 2).reshape(n_channels, -1), info)

    # チャネル情報の表示
    print("チャネル名:", raw.info['ch_names'])
    print("チャネルタイプ:", [mne.channel_type(raw.info, i) for i in range(n_channels)])

    # チャネルの詳細情報を取得
    for i, ch_name in enumerate(raw.info['ch_names']):
        ch_type = mne.channel_type(raw.info, i)
        print(f"チャネル {i+1}: 名前 = {ch_name}, タイプ = {ch_type}")

def preprocess_and_save(data_dir: str, split: str, chunk_size: int = 1000000):
    # データの読み込み
    X, subject_idxs = load_data(data_dir, split)
    
    # データの形状を取得
    n_samples, n_channels, n_times = X.shape
    
    # 形状を (n_channels, n_samples * n_times) に変形
    X_reshaped = X.permute(1, 0, 2).reshape(n_channels, -1)

    # MNE を使用して RawArray を作成
    info = mne.create_info(ch_names=[str(i) for i in range(n_channels)], sfreq=200, ch_types='eeg')
    raw = mne.io.RawArray(X_reshaped, info)

    print("raw done")

    # チャンクごとにICAを適用
    ica = mne.preprocessing.ICA(n_components=20, random_state=97)
    for start in range(0, raw.n_times, chunk_size):
        stop = min(start + chunk_size, raw.n_times)
        chunk = raw[:, start:stop][0]
        
        # チャンクを RawArray オブジェクトに変換
        chunk_raw = mne.io.RawArray(chunk, info, first_samp=start)
        
        # ICAの適用
        ica.fit(chunk_raw)
        chunk_raw = ica.apply(chunk_raw)

        # 処理後のデータを元のデータに戻す
        raw[:, start:stop] = chunk_raw[:, :][0]

    X_ica = raw.get_data()

    print("ica done")

     # 各チャネルのデータを正規化
    scaler = StandardScaler()
    X_ica = X_ica.reshape(n_channels, -1).transpose(1, 0)  # 形状を (n_samples * n_times, n_channels) に変形
    X_ica = scaler.fit_transform(X_ica)
    X_ica = X_ica.transpose(1, 0).reshape(n_channels, n_samples, n_times).transpose(1, 0, 2)  # 元の形状に戻す

    # 前処理済みデータを保存
    torch.save(torch.tensor(X_ica), os.path.join(data_dir, f"{split}_X_preprocessed.pt"))
    torch.save(subject_idxs, os.path.join(data_dir, f"{split}_subject_idxs.pt"))
    
    if split in ["train", "val"]:
        y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
        torch.save(y, os.path.join(data_dir, f"{split}_y.pt"))

# チャネル情報の確認
data_dir = "data"
split = "train"  # 例として "train" データセットを使用
X, _ = load_data(data_dir, split)
check_channel_info(X)

# 各データセットの前処理を行い、保存
for split in ["val", "train"]:
    preprocess_and_save(data_dir, split)