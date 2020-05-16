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
    parser.add_argument('input', type=str,
                        help='File containing the path to the audios to classify, one path per line.'
                             'It can also be the path to a folder with audios instead of a file.')

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
    model.eval()
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

    if os.path.isfile(args.input):
        # Read files from input file, segment them, and predict the genre
        df = pd.DataFrame(columns=['song_path'] + genrelist)
        with open(args.input, 'r') as inputfile:
            for idx, song in enumerate(inputfile.readlines()):
                song = song.strip('\n')
                segments = segment_audio(song)
                audio_spectrograms = get_spectrograms(segments)
                with t.no_grad():
                    pred = model(audio_spectrograms.to(device))
                aux = t.exp(pred)
                percentage = aux.sum(dim=0)/len(aux)
                percentage = percentage.tolist()
                df.loc[idx] = [song] + percentage
                if not args.silent:
                    print("Song '...{:.30}' is genre {:10}".format(song[-30:], genrelist[np.argmax(percentage)]))
        df.to_csv(args.output_file, index=False)
    elif os.path.isdir(args.input):
        # Read files from folder, segment them, and predict the genre
        df = pd.DataFrame(columns=['song_path'] + genrelist)
        for idx, song in enumerate(os.listdir(args.input)):
            song = os.path.join(args.input, song)
            segments = segment_audio(song)
            audio_spectrograms = get_spectrograms(segments)
            with t.no_grad():
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