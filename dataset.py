from torch.utils.data import Dataset
import numpy as np
import librosa.feature as lfeat
import librosa.core as lcore
import torch as t
import os
import pandas as pd


eps = np.finfo(float).eps


class LatinSegments(Dataset):
    """Latin music segments dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        """
        self.segments = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        genre = self.segments.iloc[idx, 1]
        arr_file = os.path.join(self.root_dir,
                                genre,
                                self.segments.iloc[idx, 0])

        segment_audio, sr = lcore.load(arr_file, sr=44100)

        if self.transform:
            segment_audio = lfeat.melspectrogram(y=segment_audio, sr=sr, hop_length=1024)

        segment_audio = t.from_numpy(segment_audio)

        return t.log(segment_audio.T.unsqueeze(0) + eps), genre


if __name__ == '__main__':
    #TODO: Dataset creation routine
    pass