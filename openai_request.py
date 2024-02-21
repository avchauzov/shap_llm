import base64
import json

import requests


with open('_data/credentials.json') as f:
	credentials = json.load(f)

openai_key = credentials.get('openai')


def encode_image(image_path):
	with open(image_path, 'rb') as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


image_path = '_data/shap_plots_scatter/sgdregressor/1.png'

base64_image = encode_image(image_path)

headers = {
		"Content-Type" : "application/json",
		"Authorization": f"Bearer {openai_key}"
		}

payload = {
		"model"     : "gpt-4-vision-preview",
		"messages"  : [
				{
						"role"   : "user",
						"content": [
								{
										"type": "text",
										"text": "Task:\n"
										        "Refine your analysis of a SHAP dependence scatter plot by detailing the linear correlation between the feature's value (x-axis) and its SHAP value (y-axis), which reveals the feature's predictive power. "
										        "Describe the linear trend and its direction to illustrate the relationship's nature. "
										        "Examine the feature's value distribution, highlighting any skewness or kurtosis that suggests a non - normal distribution. "
										        "Evaluate the impact of outliers or extreme values on the model's predictions, focusing on whether these instances correspond with significant SHAP values. "
										        "This concise evaluation should encompass the relationship's dynamics, distribution characteristics, and the influence of atypical data points on the prediction accuracy, all within a succinct 128 - word framework."
										},
								{
										"type"     : "image_url",
										"image_url":
											{
													"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"
													}
										}
								]
						}
				],
		"max_tokens": 128 * 3
		}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json().get('choices')[0].get('message').get('content'))
