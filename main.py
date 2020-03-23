from data_handler import get_tensor_spectrogram
from convolutional_neural_network import compile_3_channel_nn
from convolutional_neural_network import compile_baseline_nn

DATA_PATH = "data/"
DATA_NAME = "train-mel.arff"

def main():

	train_dataset = get_tensor_spectrogram()

	mm_cnn_model = compile_3_channel_nn()

	mm_cnn_model.fit(train_dataset, train_dataset, train_dataset, epochs = 10, batch_size=None)


