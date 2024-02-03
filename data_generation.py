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
Please provide keys and their scores for any positive outliers in the dictionary. Additionally, provide keys and their scores for any negative outliers. If a key appears more than once as an outlier, please mention its occurrence. Write an answer in a single paragraph.
'''

'''
for index, combination in enumerate(combinations):
	total = sum(abs(value) for value in combination.values())
	new_combination = {key: (value / total, 'positive') if value > 0 else (np.abs(value / total), 'negative') if value < 0 else (0.0, 'no_influence') for key, value in combination.items()}
	new_combination = dict(sorted(new_combination.items(), key=lambda value: value[1][0], reverse=True))
	combinations[index] = new_combination
'''
