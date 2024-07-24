import pandas as pd
import sys
from datasets import load_dataset
sys.path.append('../')
from lm_api import LanguageModelAPI
import prompt_library
import os
from sklearn.metrics import precision_recall_fscore_support
import re
import numpy as np

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CACHE_DIR = "/net/nfs/s2-research/joeh/lm_cache/"
DATA_PATH = "joe32140/chime-sibling-coherence"

MODEL = "gpt-4-1106-preview"
# MODEL = "gpt-3.5-turbo-0613"
openai_model = LanguageModelAPI(
    "openai",
    MODEL,
    CACHE_DIR,
    OPENAI_API_KEY)

TASK_LABEL_MAP = {
    "Task 2" :{
        "This child category is coherent with other siblings": 1,
        "This child category is NOT coherent with other siblings": 0,
    },
    "Task 3" :{
        "The claim belongs to the category": 1,
        "The claim does NOT belong to the category": 0,
    },
}

ANSWER_MAP = {
    1: "[These sibling categories are coherent]",
    0: "[These sibling categories are NOT coherent]"
}

def parse_answer(answer):
    # Split the answer into lines
    lines = answer.split('\n')

    # Define a dictionary to hold the results
    results = []

    # Iterate over the lines
    for line in lines:
        # Use a regular expression to match the category and label
        match = re.match(r'- "(.*?)": \[(.*?)\]', line)
        if match:
            # If a match was found, add it to the results
            category, label = match.groups()
            label = TASK_LABEL_MAP["Task 2"].get(label, 2)
            results.append((category, label))

    return results

ds = load_dataset(DATA_PATH)
df = pd.concat([
    ds["train"].to_pandas(),
    ds["val"].to_pandas(),
    ds["test_ID"].to_pandas(),
    ds["test_OOD"].to_pandas()]
)

df["row_id"] = df.index

test_df = df[df.split.isin(["test_ID", "test_OOD"])]

FEWSHOT = True

np.random.seed(0)
examples = []
count = 0
for i, grp_id in enumerate(test_df.sibling_group_id.unique()):
    tmp_df = test_df[test_df.sibling_group_id==grp_id]
    if count % 5 == 0:
        print(f"processing {i}th group...")
    count += 1
    parent_category = tmp_df.parent_category.iloc[0]
    sibling_categories = ", ".join(tmp_df.child_category)
    input = prompt_library.TASK2_CoT_GENERATION_PROMPT.format(
                parent_category=parent_category,
                sibling_categories=sibling_categories)
    output = openai_model.chat(input)
    examples.append((grp_id, output, tmp_df.agg_label.iloc[0]))
    

output_df = pd.DataFrame(examples, columns=["sibling_group_id", "gpt_output", "agg_label"])
output_df["gpt_label"] = output_df.gpt_output.apply( lambda x : 0 if "sibling categories are not coherent" in x.lower() else 1)

merge_df = output_df[["sibling_group_id", "gpt_output", "gpt_label"]].merge(df.drop_duplicates("sibling_group_id"), on="sibling_group_id")

all_result = precision_recall_fscore_support(1-merge_df["agg_label"], 1-merge_df["gpt_label"], average="binary")

test_id = merge_df[merge_df.split=="test_ID"]
id_result = precision_recall_fscore_support(1-test_id["agg_label"], 1-test_id["gpt_label"], average="binary")

test_ood = merge_df[merge_df.split=="test_OOD"]
ood_result = precision_recall_fscore_support(1-test_ood["agg_label"], 1-test_ood["gpt_label"], average="binary")

results = pd.DataFrame([all_result, id_result, ood_result], columns=["precision", "recall", "f1", "support"], index=["all", "id", "ood"])

print(results)