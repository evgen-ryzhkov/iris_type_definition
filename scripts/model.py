import numpy as np


class IrisClassifier:

	def train_model(self, X_train, y_train):

		# define Theta shape
		n_features = X_train.shape[1]
		n_classes = len(np.unique(y_train, axis=0))
		Theta = np.random.rand(n_features, n_classes)

		m = X_train.shape[0]
		a = 0.01    # train step
		n_iterations = 50001
		alpha = 0.1  # regularization hyperparameter

		for iteration in range(n_iterations):
			Theta_1 = Theta[1:]
			hypothesis = self._compute_hypothesis(X_train, Theta)
			cost = self._compute_cost_function(m, hypothesis, y_train, alpha, Theta_1)
			gradient = self._compute_gradient(m, X_train, y_train, hypothesis,  alpha, Theta_1, n_classes)
			if iteration % 5000 == 0:
				print('iteration / cost =', str(iteration) + ' / ' + str(cost))
			Theta = Theta - a * gradient

		return Theta

	def check_model(self, check_type, X, y, theta):
		hypothesis = self._compute_hypothesis(X, theta)
		y_predict = np.argmax(hypothesis, axis=1)
		accuracy_score = np.mean(y_predict == y)
		print(check_type+' accuracy = ', accuracy_score)


	@staticmethod
	def _compute_hypothesis(X, Theta):
		s_x = X.dot(Theta)
		exps = np.exp(s_x)
		exp_sums = np.sum(exps, axis=1, keepdims=True)
		return exps / exp_sums

	@staticmethod
	def _compute_cost_function(m, hypothesis, y, alpha, Theta_1):
		cost = (-1 / m) * np.sum(y * np.log(hypothesis) + (1 - y) * np.log(1 - hypothesis))
		cost_reg = alpha / 2 * np.sum(np.square(Theta_1))
		cost = cost + cost_reg
		return cost

	@staticmethod
	def _compute_gradient(m, X, y, hypothesis, alpha, Theta_1, n_classes):
		return (1 / m) * np.dot(X.T, hypothesis - y) + np.r_[np.zeros([1, n_classes]), alpha * Theta_1]



