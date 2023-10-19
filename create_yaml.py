# pandas
import pandas as pd
import yaml
# data
df = pd.read_csv('Somatotype data - final_data.csv', usecols=["SUBJECT", "SEX"] , encoding='cp949')
df.info()

gender = {}

for index, sex in df.values:
    if sex == 'M':
        gender['\"img_%03d\"'%int(index)] = '\"male\"'
    else:
        gender['\"img_%03d\"'%int(index)] = '\"female\"'
    
    
with open('genders.yaml', 'w') as f:
    yaml.dump(gender, f)