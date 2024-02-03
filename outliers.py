import numpy as np
from scipy.stats import median_abs_deviation


def modified_z_score_outliers(data, threshold=3.5):
	keys, values = list(data.keys()), np.array(list(data.values()))
	median = np.median(values)
	mad = np.median(np.abs(values - median))
	
	if mad == 0:
		return []
	
	modified_z = 0.6745 * (values - median) / mad
	outlier_indices = np.where(np.abs(modified_z) > threshold)[0]
	
	outliers = [keys[i] for i in outlier_indices]
	return outliers


def tukey_fences_outliers(data, lower_percentile=25, upper_percentile=75):
	keys, values = list(data.keys()), np.array(list(data.values()))
	q1, q3 = np.percentile(values, [lower_percentile, upper_percentile])
	iqr = q3 - q1
	
	inner_fence = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
	outer_fence = [q1 - 3 * iqr, q3 + 3 * iqr]
	
	outliers_inner = [key for key, value in zip(keys, values) if value < inner_fence[0] or value > inner_fence[1]]
	outliers_outer = [key for key, value in zip(keys, values) if value < outer_fence[0] or value > outer_fence[1]]
	
	return outliers_inner + outliers_outer


def percentile_based_outliers(data, lower_percentile=5, upper_percentile=95):
	keys, values = list(data.keys()), np.array(list(data.values()))
	lower_threshold = np.percentile(values, lower_percentile)
	upper_threshold = np.percentile(values, upper_percentile)
	
	outliers = [key for key, value in data.items() if value < lower_threshold or value > upper_threshold]
	return outliers


def mad(data):
	median = np.median(data)
	mad = np.median(np.abs(data - median))
	return mad


def mad_outliers(data, threshold=3.5):
	keys, values = list(data.keys()), np.array(list(data.values()))
	median = np.median(values)
	mad_value = mad(values)
	
	modified_z = 0.6745 * (values - median) / mad_value
	outlier_indices = np.where(np.abs(modified_z) > threshold)[0]
	
	outliers = [keys[i] for i in outlier_indices]
	return outliers


def huber_m_estimator(values, k=1.4826):
	mad = median_abs_deviation(values)
	huber_m = k * mad
	return huber_m


def huber_m_outliers(data, threshold=3.5):
	keys, values = list(data.keys()), list(data.values())
	huber_m = huber_m_estimator(values)
	
	outliers = [keys[i] for i, value in enumerate(values) if abs(value) > threshold * huber_m]
	return outliers
