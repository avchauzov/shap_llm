import random
import string

import numpy as np
import shap
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_random_string(length_range):
	length = random.randint(*length_range)
	alpha_char = random.choice(string.ascii_letters)
	other_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=length - 1))
	return alpha_char + other_chars


# Generate unique random strings
def generate_unique_random_strings(num_strings, length_range):
	generated_strings = set()
	while len(generated_strings) < num_strings:
		new_string = generate_random_string(length_range)
		generated_strings.add(new_string)
	return list(generated_strings)


for _ in range(128):
	n_samples = random.randint(1024, 8096)
	n_features = random.randint(4, 128)
	n_informative = random.randint(4, n_features)
	bias = random.random()
	tail_strength = random.random()
	noise = random.random()
	
	X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative, bias=bias, tail_strength=tail_strength, noise=noise)
	feature_names = generate_unique_random_strings(X.shape[1], (3, 7))
	
	test_size = random.uniform(0.05, 0.25)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
	
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	
	loss = random.choice(['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
	penalty = random.choice(['l2', 'l1', 'elasticnet', None])
	alpha = random.random()
	l1_ratio = random.random()
	fit_intercept = random.choice([True, False])
	tol = random.uniform(1e-3, 1e-1)
	
	sgd_regressor = SGDRegressor(max_iter=1024, loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, tol=tol)
	sgd_regressor.fit(X_train_scaled, y_train)
	
	explainer = shap.explainers.Linear(sgd_regressor, X_train_scaled)
	shap_values = explainer(X_test_scaled)
	
	shap_values.feature_names = feature_names
	shap_values.data = X_test
	
	mean_values = shap_values.values.mean(axis=0)
	zero_indices = np.where(mean_values == 0)[0]
	
	print(shap_values.values.shape)
	print(shap_values.base_values.shape)
	print(shap_values.data.shape)
	
	shap_values.values = np.delete(shap_values.values, zero_indices, axis=1)
	shap_values.data = np.delete(shap_values.data, zero_indices, axis=1)
	shap_values.feature_names = [feature_names[i] for i in range(len(feature_names)) if i not in zero_indices]
	
	print(shap_values.values.shape)
	print(shap_values.base_values.shape)
	print(shap_values.data.shape)
	
	shap.plots.scatter(shap_values[:, 0])
	
	'''
	shap.plots.scatter. from statistical and model developer point of view, please explain this picture. single paragraph, 128 words max.
	'''
	
	break
