import torch as t
import argparse
import os
from model import GenreClassifier
from utils import segment_audio, get_spectrograms
import pandas as pd
import numpy as np


def main():
    genrelist = ['reggaeton', 'bachata', 'salsa',
                 'merengue', 'chachacha']
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                        help='File containing the path to the audios to classify')

    parser.add_argument('-o', '--output_file', type=str, default='output.csv',
                        help='Output file with the classified audios')

    parser.add_argument('-s', '--silent', action='store_true', help='Dont print the results, only write the file')

    parser.add_argument('-m', '--model', type=str, default='model.pt',
                        help='File with the model dict_state')

    args = parser.parse_args()
    mod_state = args.model

    if not os.path.isfile(mod_state):
        raise IOError('Model state dictionary {} doesn not exist'.format(mod_state))

    # Load the model
    model = GenreClassifier()
    model.load_state_dict(t.load(mod_state))
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('---------------')
    model.to(device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(t.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(t.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(t.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

    # Read files from input file, segment them, and predict the genre
    df = pd.DataFrame(columns=['song_path'] + genrelist)
    with open(args.input_file, 'r') as inputfile:
        for idx, song in enumerate(inputfile.readlines()):
            song = song.strip('\n')
            segments = segment_audio(song)
            audio_spectrograms = get_spectrograms(segments)
            pred = model(audio_spectrograms.to(device))
            aux = t.exp(pred)
            percentage = aux.sum(dim=0)/len(aux)
            percentage = percentage.tolist()
            df.loc[idx] = [song] + percentage
            if not args.silent:
                print("Song '...{:.30}' is genre {:10}".format(song[-30:], genrelist[np.argmax(percentage)]))
    df.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()