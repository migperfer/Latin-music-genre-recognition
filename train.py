from sklearn.metrics import confusion_matrix
import torch as t
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from model import GenreClassifier
from torch.utils.data import DataLoader, random_split
from dataset import LatinSegments
import os
import argparse

def main():
    # Argument parsing:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvfile_dir", help="CSV file contaning the file_dir,genre data", type=str)
    parser.add_argument("basedir", help="Base directory for the audios", type=str)
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="Batch size used for the training. Default 32")
    parser.add_argument("-e", "--epochs", default=10, type=int, help="Number of epochs to train. Default 10")
    parser.add_argument("-m", "--model_dictionary_state", default="model.pt", type=str,
                        help="Pytorch file where to store model dictionary state. Default model.pt")

    args = parser.parse_args()

    csvfile_dir = args.csvfile_dir
    bsegmentdir = args.basedir
    if os.path.isfile(args.model_dictionary_state):
        response = input("File {} already exists. Want to override it? [y]/n: ".format(args.model_dictionary_state))
        if response == '' or response.lower() == 'y':
            m_state_dic = args.csvfile_dir
        else:
            m_state_dic = input("Type the file for model dictionary output")
    else:
        m_state_dic = args.csvfile_dir

    # Load the datasets
    daset = LatinSegments(csvfile_dir,
                          bsegmentdir,
                          transform=True)
    train_size = int(0.6 * len(daset))
    tt_aux = len(daset) - train_size
    test_size = int(0.5 * tt_aux)
    valid_size = tt_aux - test_size
    train_dataset_d, test_dataset_d, validation_dataset_d = random_split(daset, [train_size, test_size, valid_size])
    bs = args.batch_size  # Batch size
    train_dl = DataLoader(train_dataset_d, batch_size=bs)
    test_dl = DataLoader(test_dataset_d, batch_size=bs)
    val_dl = DataLoader(validation_dataset_d, batch_size=1)

    model = GenreClassifier()

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

    # The encodings
    genredict = {'reggaeton': 0, 'bachata': 1, 'salsa': 2,
                 'merengue': 3, 'chachacha': 4}

    def str_t_code(strings, diction):
        return t.tensor([diction.get(string) for string in strings]).to(device)

    loss_fun = nn.NLLLoss()  # The loss function

    # Main loop
    loss_train = []
    loss_test = []
    cmatrixes = []
    epochs = args.epochs
    try:
        for epoch in range(epochs):
            model.train()
            optimizer = Adam(model.parameters(), 1e-4 / (epoch / 4 + 1))
            # Train for this epoch
            aux = []
            print("*", "_" * 48, "*")
            print("| {:^14}| | {:^8} | | {:^7} |".format('loss', 'mini-batch(%total)', 'epoch'))
            for idx, sample in enumerate(train_dl):
                optimizer.zero_grad()
                spectrogram = sample[0]
                gcode = str_t_code(sample[1], genredict)
                predicted = model(spectrogram.to(device))
                loss = loss_fun(predicted, gcode)
                loss.backward()
                optimizer.step()
                if (idx+1) % 10 == 0:
                    print("|   {:^10.4f}   |   {:>5}({:4f}%)   |   {:^4d}   |"
                          .format(loss.item(), (idx+1), (idx+1)/len(train_dl)*100, epoch))
                aux.append(loss.item())
            loss_train.append(np.mean(aux))
            # Epoch test check
            model.eval()
            with t.no_grad():
                aux = []
                for idx, sample in enumerate(test_dl):
                    optimizer.zero_grad()
                    spectrogram = sample[0]
                    gcode = str_t_code(sample[1], genredict)
                    predicted = model(spectrogram.to(device))
                    loss = loss_fun(predicted, gcode)
                    aux.append(loss.item())
            loss_test.append(np.mean(aux))
            print("#########      FINISHED EPOCH {:^10d}     ##########".format(epoch))
            print("########      EPOCH TEST LOSS {:^10.4f}      #########".format(loss_test[-1]))
            try:
                if loss_test[-2] > loss_test[-1]:
                    t.save(model.state_dict(), m_state_dic)
            except IndexError:
                t.save(model.state_dict(), m_state_dic)

            #Epoch accuracy val
            p = []
            act = []
            inverse_dict = {val: key for key, val in genredict.items()}

            model.eval()
            with t.no_grad():
                for sample in val_dl:
                    predicted = model(sample[0].to(device))
                    predicted_g = inverse_dict[predicted.argmax(1).item()]
                    p.append(predicted_g)
                    act.append(sample[1][0])
            confusion_mat = confusion_matrix(act, p)
            ok_classifications = np.sum([confusion_mat[i][i] for i in range(confusion_mat.shape[0])])
            tot_class = np.sum(confusion_mat)
            print("########      EPOCH  ACCURACY {:^10.4f}      #########".format(ok_classifications/tot_class))
            cmatrixes.append(confusion_mat)
            np.save('loss_train.npy', loss_train)
            np.save('loss_test.npy', loss_test)
            np.save('c_matrix.npy', cmatrixes)

        print("Finished training!")
    except KeyboardInterrupt:
        print("Interrupting training")
        t.save(model.state_dict(), m_state_dic+'_interrupted')


if __name__ == '__main__':
    main()
