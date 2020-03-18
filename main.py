import data_handler as dh 
import convolutional_neural_network as cnn

DATA_PATH = "dataset/"
DATA_NAME = "train-mel.arff"


def main():

	data = dh.import_arff(DATA_PATH + DATA_NAME)

	print(data)

	data.head()

main()