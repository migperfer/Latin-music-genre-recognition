from librosa.core import load
from librosa.feature import melspectrogram
import numpy as np
from torch import Tensor, log10


eps = np.finfo(float).eps


def segment_audio(audiopath, f_duration=5, max_frag=6):
    total_audio, sr = load(audiopath, sr=44100)
    middle_audio = total_audio[int(0.15*len(total_audio)): int(-0.15*len(total_audio))]
    samples_per_segment = sr*f_duration
    total_frags = len(middle_audio) % samples_per_segment
    if max_frag != None:
        fragments = min(total_frags, max_frag)
    else:
        fragments = total_frags
    return np.array([middle_audio[i*samples_per_segment:i*samples_per_segment + samples_per_segment]
                     for i in range(fragments)])

def get_spectrograms(audio_segments):
    specs = [melspectrogram(y=segmented_audio, sr=44100, hop_length=1024).T for segmented_audio in audio_segments]
    return log10(Tensor(specs).unsqueeze(1) + eps)