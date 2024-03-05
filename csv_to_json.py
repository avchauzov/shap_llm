import json
import random
import uuid

import pandas as pd


df = pd.read_csv('_data/shap_plots_scatter/linear/description.csv', encoding='utf8')

df['id'] = [str(uuid.uuid4()) for _ in df.index]
df.rename({'file_name': 'id'}, axis=1, inplace=True)

df['conversations'] = df['description'].apply(lambda description: [{'from': 'human', 'value': '<image>\nWhat is shown in the image?'}, {'from': 'gpt', 'value': description}])
df.drop(['description'], axis=1, inplace=True)

df['_type'] = [random.choices(['train', 'test'], weights=[0.75, 0.25])[0] for _ in df.index]

train_df = df.loc[(df['_type'] == 'train')].drop(['_type'], axis=1)
test_df = df.loc[(df['_type'] == 'test')].drop(['_type'], axis=1)

train_json = train_df.to_json(orient='records')
test_json = test_df.to_json(orient='records')

with open('_data/shap_plots_scatter/linear/train.json', 'w') as json_file:
	json.dump(train_json, json_file, indent=4)

with open('_data/shap_plots_scatter/linear/test.json', 'w') as json_file:
	json.dump(test_json, json_file, indent=4)
