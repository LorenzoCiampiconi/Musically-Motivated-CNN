from scipy.io import arff
import pandas as pd


def import_arff(file):

	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])

	return df

