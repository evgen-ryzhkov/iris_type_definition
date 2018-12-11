from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IrisData:

	@staticmethod
	def get_full_data_set():
		return datasets.load_iris()

	def get_x_y_df(self):
		default_dataset = self.get_full_data_set()
		X = pd.DataFrame(default_dataset.data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
		y = pd.DataFrame(default_dataset.target, columns=['class'])
		united_df = X.join(y)
		return united_df

	@staticmethod
	def search_correlation(dataset):
		corr_matrix = dataset.corr()
		print('Correlation to iris class:\n', corr_matrix['class'].sort_values(ascending=False))

	@staticmethod
	def show_data_distribution(iris_df):
		plt.figure(figsize=(10, 4))
		iris_class_0 = iris_df.loc[iris_df['class'] == 0]
		plt.plot(iris_class_0['petal length'], iris_class_0['petal width'], 'g^')
		iris_class_1 = iris_df.loc[iris_df['class'] == 1]
		plt.plot(iris_class_1['petal length'], iris_class_1['petal width'], 'bs')
		iris_class_2= iris_df.loc[iris_df['class'] == 2]
		plt.plot(iris_class_2['petal length'], iris_class_2['petal width'], 'yo')
		plt.xlabel("Petal length", fontsize=14)
		plt.ylabel("Petal width", fontsize=14)
		plt.axis([0, 7, 0, 3.5])
		plt.show()

	def get_split_data_set(self):
		default_dataset = self.get_full_data_set()
		X_prepared, n = self._prepare_x_data_set(default_dataset)
		y = default_dataset.target

		# split the data set
		test_set_size = int(n * 0.2)
		validation_set_size = int(test_set_size)
		train_set_size = n - (test_set_size + validation_set_size)

		np.random.seed(2042)
		mixed_indixes = np.random.permutation(n)
		train_indexes = mixed_indixes[:train_set_size]
		validation_indexes = mixed_indixes[train_set_size:(train_set_size+test_set_size)]
		test_indexes = mixed_indixes[-test_set_size:]

		X_train = X_prepared[train_indexes]
		y_train_temp = y[train_indexes]
		y_train = self._prepare_y_data_set(y_train_temp)
		X_validation = X_prepared[validation_indexes]
		y_validation = y[validation_indexes]
		X_test = X_prepared[test_indexes]
		y_test = y[test_indexes]

		return X_train, y_train, X_validation, y_validation, X_test, y_test

	@staticmethod
	def _prepare_x_data_set(default_dataset):
		# add bias column (x_0 = 1)
		X = default_dataset.data[:, (2, 3)]  # petal length, petal width as the most correlationid
		n = len(X)
		bias = np.ones([n, 1])
		X_with_bias = np.concatenate((bias, X), axis=1)
		return X_with_bias, n

	@staticmethod
	def _prepare_y_data_set(y_dataset):
		# getting every y instance as a one-hot vector
		# 0 class = [1, 0, 0]
		# 1 class = [0, 1, 0]
		# 2 class = [0, 0, 1]
		# y = default_dataset.target
		m = y_dataset.shape[0]
		n_classes = y_dataset.max() + 1
		Y_hot = np.zeros((m, n_classes))
		for idx, y_i in np.ndenumerate(y_dataset):
			if y_i == 0:
				Y_hot[idx, 0] = 1
			elif y_i == 1:
				Y_hot[idx, 1] = 1
			else:
				Y_hot[idx, 2] = 1
		return Y_hot
