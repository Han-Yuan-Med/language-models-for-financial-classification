import transformers
import torch
from transformers import AutoTokenizer
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import accelerate
import pandas as pd
import numpy as np
import json
from scipy.stats import chisquare

german_text = pd.read_csv('German_text.csv')
data_good = german_text.iloc[np.where(german_text.iloc[:, 1] == 'Good')[0][:300]]
data_bad = german_text.iloc[np.where(german_text.iloc[:, 1] == 'Bad')[0]]

model_id = "microsoft/Phi-3.5-mini-instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.half,
    device_map="cuda",
    top_k=1
)


def classify_explanation(textual_information):
    prompt_explanation = f"Assess the creditworthiness of a customer using the following attributes for financial status. Respond with the final decision of either 'good credit' or 'bad credit' in the first line. Respond with the explanation on the final decision in the second line. And the attributes are as follows: {textual_information}. Response:"
    messages = [{"role": "user", "content": prompt_explanation}]

    model_output = pipe(
        messages,
        max_new_tokens=1024,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    return model_output


model_output_content = []
for i in tqdm(range(len(data_good))):
    output = classify_explanation(data_good.iloc[i, 0])
    model_output_content.append([data_good.iloc[i, 0], output[0]['generated_text'][-1]['content']])

print('LLMs finished summary on the 300 samples of good credit')
model_output_content = pd.DataFrame(model_output_content, columns=['Model input', 'Model output'])
model_output_content.to_excel('Phi_good_credit_accuracy.xlsx', index=False)

model_output_content = pd.read_excel('Phi_good_credit_accuracy.xlsx')
good_correct_id = []
good_wrong_id = []
for i in range(len(model_output_content)):
    print(model_output_content.iloc[i, 1].split(';')[0])
    if model_output_content.iloc[i, 1].split(';')[0] == ' good credit':
        good_correct_id.append(i)
    if model_output_content.iloc[i, 1].split(';')[0] == ' bad credit':
        good_wrong_id.append(i)

model_output_content = []
for i in tqdm(range(len(data_bad))):
    output = classify_explanation(data_bad.iloc[i, 0])
    model_output_content.append([data_bad.iloc[i, 0], output[0]['generated_text'][-1]['content']])

print('LLMs finished summary on all samples of bad credit')
model_output_content = pd.DataFrame(model_output_content, columns=['Model input', 'Model output'])
model_output_content.to_excel('Phi_bad_credit_accuracy.xlsx', index=False)

model_output_content = pd.read_excel('Phi_bad_credit_accuracy.xlsx')
bad_correct_id = []
bad_wrong_id = []
for i in range(len(model_output_content)):
    print(model_output_content.iloc[i, 1].split(';')[0])
    if model_output_content.iloc[i, 1].split(';')[0] == ' bad credit':
        bad_correct_id.append(i)
    if model_output_content.iloc[i, 1].split(';')[0] == ' good credit':
        bad_wrong_id.append(i)

Accuracy = (len(good_correct_id) + len(bad_correct_id)) / (len(good_correct_id) + len(bad_correct_id) +
                                                           len(good_wrong_id) + len(bad_wrong_id))
Precision = len(bad_correct_id) / (len(bad_correct_id) + len(good_wrong_id))
Recall = len(bad_correct_id) / (len(bad_correct_id) + len(bad_wrong_id))
Sensitivity = len(bad_correct_id) / (len(bad_correct_id) + len(bad_wrong_id))
Specificity = len(good_correct_id) / (len(good_correct_id) + len(good_wrong_id))
F1 = 2 * Precision * Recall / (Precision + Recall)
Cost = 5 * len(bad_wrong_id) + 1 * len(good_wrong_id)
