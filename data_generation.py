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

'''
Positive Influences:
The most positively influential feature is f with a SHAP value of 0.1021, indicating that higher values of f are likely to increase the prediction of the target variable.
Other positive contributors include w (0.0782), i (0.0647), x (0.0444), r (0.0391), g (0.0367), e (0.0175), and k (0.0132). These features, when present or increased in value, tend to push the model's predictions higher.
The distribution of positive features shows a wide range of influence, from f having the strongest positive impact to k with the least. This suggests a variety of features play a role in positively influencing the target, but their degree of impact varies significantly.
Negative Influences:
The feature with the most negative influence on the target variable is s with a SHAP value of 0.1112, suggesting that increases in s lead to lower predictions of the target variable.
Other negatively influencing features include p (0.0903), z (0.0795), l (0.0752), u (0.0667), a (0.0614), d (0.0302), y (0.0300), n (0.0288), v (0.0162), and b (0.0149). These features, when their values increase, generally contribute to lowering the model's output.
The negative features exhibit a similar range of influence as the positive ones, with s being the most impactful in reducing the target variable's prediction and b having the least negative impact. This highlights a balance in how different features can pull the prediction in opposite directions.
Notable Clusters and Outliers:
Positive and negative influences are relatively balanced in terms of the number of features contributing to each side, but the magnitude of their influence varies. Features like s, f, p, and z are more dominant in their respective directions.
There's a notable cluster of features (u, l, z, p) with negative influences that are closely grouped together in terms of their SHAP values, suggesting these features might be related or have similar effects on the model's predictions.
As for outliers, f stands out as the most significant positive outlier, while s is the most significant negative outlier. These features are especially important for understanding the model's behavior since they have the strongest individual impacts.
Summary:
This analysis of SHAP values indicates a balanced but varied influence of features on the target variable, with both positive and negative contributors. The significant spread in the magnitude of SHAP values highlights the importance of considering each feature's individual impact on the model's predictions. Identifying clusters and outliers among these features can also provide insights into how different aspects of the data interact with the model, potentially guiding further data analysis or feature engineering efforts to improve model performance.
'''
