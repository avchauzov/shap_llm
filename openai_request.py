import base64
import json
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


with open('_data/credentials.json') as f:
	credentials = json.load(f)

openai_key = credentials.get('openai')

HEADER = {
		'Content-Type' : 'application/json',
		'Authorization': f'Bearer {openai_key}'
		}

OUTPUT_SIZE = 128

PROMPT = ' '.join(
		['Task:\n',
		 'Refine your analysis of a SHAP dependence scatter plot by detailing the linear correlation between the feature\'s value (x-axis) and its SHAP value (y-axis), which reveals the feature\'s predictive power. ',
		 'Describe the linear trend and its direction to illustrate the relationship\'s nature. ',
		 'Examine the feature\'s value distribution, highlighting any skewness or kurtosis that suggests a non - normal distribution. ',
		 'Evaluate the impact of outliers or extreme values on the model\'s predictions, focusing on whether these instances correspond with significant SHAP values. ',
		 f'This concise evaluation should encompass the relationship\'s dynamics, distribution characteristics, and the influence of atypical data points on the prediction accuracy, all within a succinct {OUTPUT_SIZE} - word framework.']
		)

def encode_image(image_path):
	with open(image_path, 'rb') as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


output = []
folder_path = Path('_data/shap_plots_scatter/linear/images')
for file_path in tqdm(folder_path.glob('*.png')):
	image = Image.open(file_path)
	width, height = image.size
	
	if min(width, height) > 512:
		continue
	
	image = encode_image(file_path)
	
	request = {
			'model'     : 'gpt-4-vision-preview',
			'messages'  : [
					{
							'role'   : 'user',
							'content': [
									{
											'type': 'text',
											'text': PROMPT
											},
									{
											'type'     : 'image_url',
											'image_url':
												{
														'url'   : f'data:image/jpeg;base64,{image}',
														'detail': 'low'
														}
											}
									]
							}
					],
			'max_tokens': OUTPUT_SIZE * 3
			}
	
	response = requests.post('https://api.openai.com/v1/chat/completions', headers=HEADER, json=request)
	
	try:
		response = response.json().get('choices')[0].get('message').get('content')
		output.append((str(file_path), response))
	
	except Exception as _:
		continue

# break

df = pd.DataFrame(output, columns=['file_path', 'description'])
df.to_csv('_data/shap_plots_scatter/linear/description.csv', index=False)

'''
https://github.com/doccano/doccano
'''
