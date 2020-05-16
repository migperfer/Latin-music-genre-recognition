# Latin-music-genre-recognition
Deep Learning for _Salsa/Bachata/Merengue/Chachacha/Reggaeton_ music recognition

This repository contains an implementation of the _CNN Max Pooling_ network described [here](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2045.pdf) with a few differences:
- Instead of using 30 seconds audio to train I take segments of 5 seconds long at `sr=44100`. This makes the model very 
small, about 30k parameters.
- The number of points taken for the fourier transform is `2048` and the hop size is `1024`. The log10 of the melscale values are taken.
- For prediction the model takes 6 segments (of 5 seconds) from the middle section of the song, and use it as a 
minibatch. Selection of the genre is done according to the mean probability for each genre across the 6 audios fragments.
- The number of output classes are now 5.

The files presented in this repository are:
- `model.py`: This is the script with the Network implemented for predict the genres.
- `dataset.py`: Contain the _LatinSegments_ object which comes from `pytorch.utils.data.Dataset`
- `train.py`: This is the script used to train the model.
- `predict.py`: This script takes a trained model and predict the genre of music stored in your computer.
- `model.pt`: The pre-trained model. 
- `utils.py`: Some different functions used for analysis.

Unfortunately the database I used to train the model can't be shared due to legal reasons. However I provide the state_dict
for the trained in the `model.pt` file. I reached an accuracy of 90.1% for a 3-fold cross validation. 

## Making predictions with the model
To classify audios stored in your computer use the `predict.py`. You can provide the songs as the first argument in two ways:
- A file with the paths to the audios. One song per line:
```text
<path/to/song1.wav>
<path/to/song2.mp3>
<path/to/song3.m4a>
...
```
- A path to a directory containing the audios to classify. 

You will see the output of the predictions in the console:

```console
Song '...ut_file\02. La chula Linda.m4a' is genre chachacha 
Song '...input_file\02. La Novia.mp3' is genre merengue  
Song '...ile\02. Por Favor (Please).m4a' is genre chachacha 
Song '...lor que pena – Melina Leon.mp3' is genre merengue  
```
But it will also store the results in a csv file with the name `output.csv`. You can suppress the output of the console by using the `-s, --silent` option.
You can also change the output csv file with the `-o, --output_file` option. By default the dict_state used for the model is the one on the `model.py`.
Call the  `predict.py` script with the `-h` option for more details.

The output csv file is slightly different to the console output. This file contains the probability assigned to the songs of belonging to each of the 5 genres:
```csv
song_path,reggaeton,bachata,salsa,merengue,chachacha
path/to/song1.wav,0.01461903564631939,0.0327363982796669,0.07929264008998871,0.004370272159576416,0.8689815998077393
path/to/song2.wav,0.013173307292163372,0.011491022072732449,0.01588541641831398,0.010149138048291206,0.9493011236190796
```
As you can see the sum of each row returns 1.
## Training your own model
Use the `train.py` script to train a model with your own data. The script takes two mandatory arguments:
- A csv file containing the paths to the segments of 5 seconds along with the genre of the segment:
```csv
file,genre
path/to/audio1.wav,genre_of_the_segment1
path/to/audio2.wav,genre_of_the_segment2
path/to/audio3.wav,genre_of_the_segment3
```

- The base directory for the audios. If the audios paths are already written in absolute paths just write `/` for this argument.
Other parameters allow you to choose the batch size used for training, the number of epochs and the file to store the model dict_state.
 Call the `train.py` script with the `-h` option for more details.
 
 This script will also create 3 _numpy_ files:
 * loss_train.npy: Contain the mean loss in training for each epoch.
 * loss_test.npy: Contain the mean loss in test for each epoch.
 * c_matrix.npy: Contain the confusion matrices for each epoch.
 
If you want to check the confusion matrices, this will be helpful:
```python
# The encodings
genredict = {'reggaeton': 0, 'bachata': 1, 'salsa': 2,
             'merengue': 3, 'chachacha': 4}
``` 
### Requirements
- Librosa
- Pytorch
- Numpy
- Pandas
- [Optional] Cudatoolkit to speed up your training/prediction process.

