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
DATA_PATH = "joe32140/chime-claim-category"

# MODEL = "gpt-3.5-turbo-0613"
MODEL = "gpt-4-1106-preview"
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


dataset = load_dataset(DATA_PATH)
df = pd.concat([dataset["train"].to_pandas(), dataset["val"].to_pandas(), dataset["test_ID"].to_pandas(), dataset["test_OOD"].to_pandas()])

df["row_id"] = df.index

test_df = df[df.split.isin(["test_ID", "test_OOD"])]

np.random.seed(0)
examples = []
for i, row in test_df.iterrows():
    # if i % 50 == 0:
    print("\rprocessing row", i, end="")
    input = prompt_library.TASK3_CoT_GENERATION_PROMPT.format(row.claim, row.category)
    
    # retry if error 
    while True:
        try:
            output = openai_model.chat(input)
            break
        except:
            continue
    
    examples.append((row.review_pmid, row.hierarchy_id, row.claim, row.category, row.row_id, output))


output_df = pd.DataFrame(examples, columns=["review_pmid", "hierarchy_id","claim", "category", "row_id", "gpt_output"])
output_df["gpt_label"] = output_df.gpt_output.apply( lambda x : 1 if "claim belongs to the category" in x.lower() else 0)

merge_df = df.merge(output_df[["gpt_label", "gpt_output", "row_id"]], on="row_id")


# we need to aggregate the label of claim vs category based on the node path
agg_label = []
for group in merge_df.groupby(["review_pmid", "hierarchy_id", "claim"]):
    review_pmid, hierarchy_id, claim = group[0]
    grp_df = group[1]
    for i, row in grp_df.iterrows():
        node_path = list(map(int, row.node_path.split(",")))
        label = grp_df[
            grp_df.node_id.isin(node_path)
        ].gpt_label.mean() == 1
        agg_label.append({
            "review_pmid": review_pmid,
            "hierarchy_id": hierarchy_id,
            "claim": claim,
            "node_id": row.node_id,
            "gpt_agg_label": int(label)
        })
agg_label_df = pd.DataFrame(agg_label)
merge_df = merge_df.merge(agg_label_df, on=["review_pmid", "hierarchy_id", "claim", "node_id"], how="left")

all_result = precision_recall_fscore_support(merge_df["agg_label"], merge_df["gpt_agg_label"], average="binary")

test_id = merge_df[merge_df.split=="test_ID"]
id_result = precision_recall_fscore_support(test_id["agg_label"], test_id["gpt_agg_label"], average="binary")

test_ood = merge_df[merge_df.split=="test_OOD"]
ood_result = precision_recall_fscore_support(test_ood["agg_label"], test_ood["gpt_agg_label"], average="binary")

results = pd.DataFrame([all_result, id_result, ood_result], columns=["precision", "recall", "f1", "support"], index=["all", "id", "ood"])

print(results)