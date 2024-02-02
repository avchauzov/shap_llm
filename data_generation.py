import random
import string

import numpy as np


def generate_random_dict():
	num_elements = random.randint(2, 128)
	random_dict = {}
	
	keys = np.random.choice(list(string.ascii_lowercase), num_elements)
	for key in keys:
		value = random.uniform(-1, 1)
		random_dict[key] = value
	
	return random_dict


combinations = [generate_random_dict() for _ in range(128)]

for index, combination in enumerate(combinations):
	total = sum(abs(value) for value in combination.values())
	new_combination = {key: (value / total, 'positive') if value > 0 else (np.abs(value / total), 'negative') if value < 0 else (0.0, 'no_influence') for key, value in combination.items()}
	new_combination = dict(sorted(new_combination.items(), key=lambda value: value[1][0], reverse=True))
	combinations[index] = new_combination

for i, combination in enumerate(combinations[:5], 1):
	print(f"Combination {i}: {combination}")

'''
Can you analyze these SHAP values for feature influences on the target variable and provide a summary highlighting the positive and negative factors, their distribution, and any notable clusters or outliers?
'''
