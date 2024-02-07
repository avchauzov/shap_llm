import random
import string
from collections import Counter

import numpy as np

from outliers import huber_m_outliers, mad_outliers, modified_z_score_outliers, percentile_based_outliers, tukey_fences_outliers


def generate_random_dict():
	num_elements = random.randint(32, 128)
	random_dict = {}
	
	keys = np.random.choice(list(string.ascii_lowercase), num_elements)
	for key in keys:
		value = random.uniform(-1, 1)
		random_dict[key] = value
	
	random_dict = dict(sorted(random_dict.items(), key=lambda item: item[1], reverse=True))
	return random_dict


combinations = [generate_random_dict() for _ in range(128)]

outliers = []
for index, combination in enumerate(combinations):
	outlier = modified_z_score_outliers(combination)
	outlier.extend(tukey_fences_outliers(combination))
	outlier.extend(percentile_based_outliers(combination))
	outlier.extend(mad_outliers(combination))
	outlier.extend(huber_m_outliers(combination))
	
	outlier = Counter(outlier)
	outlier = sorted(outlier.items(), key=lambda item: (-item[1], item[0]))
	outlier = dict(outlier)
	
	positive_outlier = {key: value for key, value in outlier.items() if combination.get(key) > 0}
	negative_outlier = {key: value for key, value in outlier.items() if combination.get(key) < 0}
	
	outliers.append([positive_outlier, negative_outlier])

for i, (combination, outlier) in enumerate(zip(combinations, outliers)):
	combination = {key: np.round(value, 2) for key, value in combination.items()}
	print(f'Combination: {combination}')
	
	if len(list(outlier[0].values())) > 0:
		positive_outlier = '; '.join([f'"{key}"' if value == 1 else f'"{key}" (occurred {value} times)' for key, value in outlier[0].items()])
		print(f'Positive Outliers: {positive_outlier}')
	
	else:
		print(f'Positive Outliers: not found')
	
	if len(list(outlier[1].values())) > 0:
		negative_outlier = '; '.join([f'"{key}"' if value == 1 else f'"{key}" (occurred {value} times)' for key, value in outlier[1].items()])
		print(f'Negative Outliers: {negative_outlier}')
	
	else:
		print(f'Negative Outliers: not found')
	
	print()

'''
Request:
"""
Combination: {'c': 0.69, 'v': 0.63, 'r': 0.61, 'g': 0.58, 'l': 0.49, 'f': 0.39, 'b': 0.35, 'a': 0.31, 'm': 0.27, 'k': 0.19, 's': 0.18, 'o': 0.17, 'i': 0.02, 'd': -0.17, 'p': -0.36, 'n': -0.48, 'e': -0.51, 'w': -0.66, 'z': -0.77, 'h': -0.78, 'y': -0.78, 'u': -0.92, 'j': -0.92, 'x': -0.93, 'q': -0.96, 't': -0.99}
Positive Outliers: "c"; "v"
Negative Outliers: "q"; "t"
"""

Response:
For this model, we see two potential features that significantly positively influence the output: "c" and "v", with scores of 0.69 and 0.63, respectively. On the other side, features "q" and "t" pull the target down with scores of 0.99 and 0.96.

Request:
"""
Combination: {'p': 0.84, 'k': 0.83, 'h': 0.77, 'a': 0.77, 'v': 0.57, 'j': 0.45, 'd': 0.41, 'i': 0.36, 'z': 0.32, 'm': 0.2, 'l': -0.01, 'r': -0.1, 's': -0.14, 'n': -0.14, 't': -0.21, 'g': -0.23, 'e': -0.31, 'x': -0.37, 'y': -0.39, 'b': -0.53, 'o': -0.53, 'w': -0.62, 'u': -0.65, 'q': -0.8, 'f': -0.89, 'c': -0.93}
Positive Outliers: "k"; "p"
Negative Outliers: "c"; "f"
"""
'''

'''
for index, combination in enumerate(combinations):
	total = sum(abs(value) for value in combination.values())
	new_combination = {key: (value / total, 'positive') if value > 0 else (np.abs(value / total), 'negative') if value < 0 else (0.0, 'no_influence') for key, value in combination.items()}
	new_combination = dict(sorted(new_combination.items(), key=lambda value: value[1][0], reverse=True))
	combinations[index] = new_combination
'''
