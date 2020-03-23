from scipy.io import arff
import pandas as pd
import librosa as lib
from scipy.signal import windows as win
from collections import namedtuple
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os, glob

window = win.blackmanharris(2048)
folder_length = 89
root_folder = '../Musically-Motivated-CNN/data/'
steps_back = '../../'

LabeledChunk = namedtuple('Chunk', ['label', 'spectrogram'])
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
chunk_len = 80

def extract_labeled_spectrogram():

	labeled_dataset = []

	print('starting feature exctration')

	for genre in genres:
		os.chdir(root_folder + genre)
		for file in glob.glob('*.au'):

			loaded_audio, sr = lib.load(file)
			spectrogram = lib.feature.melspectrogram(loaded_audio, sr=sr, window = window,
													 hop_length = 1024, n_mels = 40)

			labeled_dataset += [LabeledChunk(genre, spectrogram[:, i * chunk_len: (i + 1) * chunk_len])
								for i in range(int(len(spectrogram[0,:]) / chunk_len))]

		print(genre)

		os.chdir(steps_back)


	return labeled_dataset


def get_tensor_spectrogram():

	# get feature from data
	labeled_dataset = extract_labeled_spectrogram()

	# split label and feature
	targets = [s.label for s in labeled_dataset]
	values = [s.spectrogram for s in labeled_dataset]

	# targets to one oh encoding
	label_encoder = LabelEncoder()

	targets = label_encoder.fit_transform(targets)
	targets = to_categorical(targets)

	tensor_dataset = tf.data.Dataset.from_tensor_slices((targets, values))

	for feat, targ in tensor_dataset.take(5):
		print('Features: {}, Target: {}'.format(feat, targ))

	train_dataset = tensor_dataset.shuffle(len(labeled_dataset)).batch(1)

	return train_dataset


def from_arff_to_dataframe(file):

	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])

	return df

