## use code below to generation JSON formated data for submission.
## The code below takes in a dataframe with two columns: image_name and PCI

import pandas as pd
import json


# df: should have two columns - image_name and PCI
def gen_submit(df):
    out_json = []
    for _, results in df.iterrows():
        out_json.append({results['image_name']: results['PCI']})
    with open('submission.json', 'w') as f:
        json.dump(out_json, f)


