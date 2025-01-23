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

german = pd.read_csv('German.csv')
data_good = german.iloc[np.where(german.iloc[:, -1] == 1)[0][:300]]
data_bad = german.iloc[np.where(german.iloc[:, -1] == 2)[0]]

model_id = "microsoft/Phi-3.5-mini-instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.half,
    device_map="cuda",
    top_k=1
)

dic = {
    'A11': "Status of existing checking account is less than 0 DM; ",
    'A12': "Status of existing checking account is between 0 DM and 200 DM; ",
    'A13': "Status of existing checking account is larger than 200 DM or salary assignments for at least 1 year; ",
    'A14': "Status of existing checking account is no checking account; ",
    'A30': 'Credit history is no credits taken or all credits paid back duly; ',
    'A31': 'Credit history is all credits at this bank paid back duly; ',
    'A32': 'Credit history is existing credits paid back duly till now; ',
    'A33': 'Credit history is delay in paying off in the past; ',
    'A34': 'Credit history is critical account or other credits existing (not at this bank); ',
    'A40': 'Purpose is car (new); ',
    'A41': 'Purpose is car (used); ',
    'A42': 'Purpose is furniture/equipment; ',
    'A43': 'Purpose is radio/television; ',
    'A44': 'Purpose is domestic appliances; ',
    'A45': 'Purpose is repairs; ',
    'A46': 'Purpose is education; ',
    'A47': 'Purpose is (vacation - does not exist?); ',
    'A48': 'Purpose is retraining; ',
    'A49': 'Purpose is business; ',
    'A410': 'Purpose is others; ',
    'A61': 'Savings account/bonds is less than 100 DM; ',
    'A62': 'Savings account/bonds is between 100 DM and 500 DM; ',
    'A63': 'Savings account/bonds is between 500 DM and 1000 DM; ',
    'A64': 'Savings account/bonds is larger than 1000 DM; ',
    'A65': 'Savings account/bonds is unknown/ no savings account; ',
    'A71': 'Present employment since is unemployed; ',
    'A72': 'Present employment since is shorter than 1 year; ',
    'A73': 'Present employment since is between 1 year to 4 years; ',
    'A74': 'Present employment since is between 4 years to 7 years; ',
    'A75': 'Present employment since is longer than 7 years; ',
    'A91': 'Personal status and sex is divorced/separated male; ',
    'A92': 'Personal status and sex is divorced/separated/married female; ',
    'A93': 'Personal status and sex is single male; ',
    'A94': 'Personal status and sex is married/widowed male; ',
    'A95': 'Personal status and sex is single female; ',
    'A101': 'Other debtors or guarantors is none; ',
    'A102': 'Other debtors or guarantors is co-applicant; ',
    'A103': 'Other debtors or guarantors is guarantor; ',
    'A121': 'Property is real estate; ',
    'A122': 'Property is building society savings agreement/life insurance; ',
    'A123': 'Property is car or other; ',
    'A124': 'Property is unknown or no property; ',
    'A141': 'Other installment plans is bank; ',
    'A142': 'Other installment plans is stores; ',
    'A143': 'Other installment plans is none; ',
    'A151': 'Housing is rent; ',
    'A152': 'Housing is own; ',
    'A153': 'Housing is for free; ',
    'A171': 'Job is unemployed or unskilled  - non-resident; ',
    'A172': 'Job is unskilled - resident; ',
    'A173': 'Job is skilled employee or official; ',
    'A174': 'Job is management or self-employed or highly qualified employee or officer; ',
    'A191': 'Telephone is none; ',
    'A192': 'Telephone is yes, registered under the customers name; ',
    'A201': 'foreign worker is yes.',
    'A202': 'foreign worker is no.',
}


def classify_explanation(tabular_feature):
    textual_information = ''
    for j in range(20):
        if j in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]:
            textual_information += dic[tabular_feature.iloc[j]]
        else:
            if j == 1:
                textual_information += f"Duration in month is {tabular_feature.iloc[1]}; "
            if j == 4:
                textual_information += f"Credit amount is {tabular_feature.iloc[4]}; "
            if j == 7:
                textual_information += f"Installment rate in percentage of disposable income is {tabular_feature.iloc[7]}; "
            if j == 10:
                textual_information += f"Present residence since is {tabular_feature.iloc[10]}; "
            if j == 12:
                textual_information += f"Age in years is {tabular_feature.iloc[12]}; "
            if j == 15:
                textual_information += f"Number of existing credits at this bank is {tabular_feature.iloc[15]}; "
            if j == 17:
                textual_information += f"Number of people being liable to provide maintenance for is {tabular_feature.iloc[17]}; "
    prompt_explanation = f"Assess the creditworthiness of a customer using the following attributes for financial status. Respond with the final decision of either 'good credit' or 'bad credit' in the first line. Respond with the explanation on the final decision in the second line. And the attributes are as follows: {textual_information}. Response:"
    messages = [{"role": "user", "content": prompt_explanation}]
    model_output = pipe(
        messages,
        max_new_tokens=1024,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    return textual_information, model_output


model_output_content = []
for i in tqdm(range(len(data_good))):
    text, output = classify_explanation(data_good.iloc[i, :20])
    model_output_content.append([text, output[0]['generated_text'][-1]['content']])

print('LLMs finished summary on the 300 samples of good credit')
model_output_content = pd.DataFrame(model_output_content, columns=['Model input', 'Model output'])
model_output_content.to_excel('Phi_good_credit_accuracy_original.xlsx', index=False)

model_output_content = pd.read_excel('Phi_good_credit_accuracy_original.xlsx')
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
    text, output = classify_explanation(data_bad.iloc[i, :20])
    model_output_content.append([text, output[0]['generated_text'][-1]['content']])

print('LLMs finished summary on all samples of bad credit')
model_output_content = pd.DataFrame(model_output_content, columns=['Model input', 'Model output'])
model_output_content.to_excel('Phi_bad_credit_accuracy_original.xlsx', index=False)

model_output_content = pd.read_excel('Phi_bad_credit_accuracy_original.xlsx')
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
