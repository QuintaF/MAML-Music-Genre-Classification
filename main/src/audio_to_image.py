#directory change ->  ...\main\src
import os
file_path = os.path.dirname(__file__)
os.chdir(file_path)

#modules
import numpy as np
import matplotlib.pyplot as plt
import librosa

#global
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def get_filenames():
    '''
    Retrieves filenames from the folder

    Returns
        list of filenames
    '''
    filenames = {}
    # open audio files directory
    path = "../audio_dataset/"
    for genre in os.listdir(path):
        filenames[genre] = []
        for song in os.listdir(path + genre):
            filenames[genre].append(path + genre + '/' + song)

    return filenames


def mel_spectrogram(song, genre, data):
    '''
    Computes the mel spectrogram and saves an image

    Args
        song: audio file
        genre: song genre
        data: audio file data
    '''

    name = song.rsplit('/',1)[1]
    plt.figure(name,figsize=(17,5))
    mel = librosa.feature.melspectrogram(y = data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH ,dtype="float64")
    mel_db = librosa.power_to_db(mel, ref=np.max)

    librosa.display.specshow(mel_db, y_axis="mel", x_axis="time")
    plt.colorbar(format="%+2.0f dB")
    plt.set_cmap("magma")

    plt.savefig(f"../dataset/mel_spectrograms/{genre}/" + name[:-3] + "jpg")
    plt.close()


def extraction_pipeline():
    filenames = get_filenames()

    for genre, paths in filenames.items():
        os.makedirs(f"../dataset/mel_spectrograms/{genre}/", exist_ok=True)
        for song in paths:
            data, _ = librosa.load(song)
            mel_spectrogram(song, genre, data)
    
    return 0


if __name__ == '__main__':
    extraction_pipeline()