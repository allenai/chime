
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import os

# fix seeding for pytorch and huggingface
import torch
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

metric = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


# Load the tokenizer, model, and data collator
MODEL_SIZE = "base"
MODEL_NAME = f"google/flan-t5-{MODEL_SIZE}"
OUTPUT_DIR = f"./task3_flant5_{MODEL_SIZE}/"
DATA_PATH = "joe32140/chime-claim-category"

# create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# We prefix our tasks with "answer the question"
prefix = "Please answer this question: Does the claim belong to the category?"

# Define the preprocessing function
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [
      f"{prefix} Claim: {claim} Category: {category}" 
      for claim, category in zip(examples["claim"], examples["category"])]
   model_inputs = tokenizer(inputs, max_length=128, truncation=True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text=list(map(str, examples["human_label"])),
                      max_length=8,
                      padding=True,
                      truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [int(pred.strip()) for pred in preds]
    labels = [int(label.strip()) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    print("Number of predictions: ", len(decoded_preds))
    # compute f1, precision, recall, and accuracy 
    result = {}
    result.update(metric.compute(predictions=decoded_preds, references=decoded_labels))
    result.update(f1.compute(predictions=decoded_preds, references=decoded_labels, average='binary'))
    result.update(precision.compute(predictions=decoded_preds, references=decoded_labels, average='binary'))
    result.update(recall.compute(predictions=decoded_preds, references=decoded_labels, average='binary'))

    return result


tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

dataset = load_dataset(DATA_PATH)
df = pd.concat([dataset["train"].to_pandas(), dataset["val"].to_pandas(), dataset["test_ID"].to_pandas(), dataset["test_OOD"].to_pandas()])

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['claim', 'category', 'human_label', 'split'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=3e-4,

    num_train_epochs=2,
    # logging & evaluation strategies
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="epoch", 
    # logging_steps=1000,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=False,
    # metric_for_best_model="overall_f1",
    # push to hub parameters
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    compute_metrics=compute_metrics,
)

# Start training 
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR + "model")

# # Load model to trainer
# model = T5ForConditionalGeneration.from_pretrained(OUTPUT_DIR + "model")
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["val"],
#     compute_metrics=compute_metrics,
# )

# concat two datasets test_ID and test_OOD
from datasets import concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support
test_dataset = concatenate_datasets([tokenized_dataset["test_ID"], tokenized_dataset["test_OOD"]])
test_results = trainer.predict(test_dataset)
preds, labels = test_results[0], test_results[1]
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

test_df = pd.concat([df[df.split=="test_ID"], df[df.split=="test_OOD"]])
test_df["flan_t5_pred"] = decoded_preds
test_df["flan_t5_label"] = decoded_labels
test_df["flan_t5_pred"] = test_df["flan_t5_pred"].astype(int)
test_df["flan_t5_label"] = test_df["flan_t5_label"].astype(int)

# we need to aggregate the label of claim vs category based on the node path
agg_label = []
for group in test_df.groupby(["review_pmid", "hierarchy_id", "claim"]):
    review_pmid, hierarchy_id, claim = group[0]
    grp_df = group[1]
    for i, row in grp_df.iterrows():
        node_path = list(map(int, row.node_path.split(",")))
        label = grp_df[
            grp_df.node_id.isin(node_path)
        ].flan_t5_pred.mean() == 1
        agg_label.append({
            "review_pmid": review_pmid,
            "hierarchy_id": hierarchy_id,
            "claim": claim,
            "node_id": row.node_id,
            "flant5_agg_label": int(label)
        })
agg_label_df = pd.DataFrame(agg_label)
test_df = test_df.merge(agg_label_df, on=["review_pmid", "hierarchy_id", "claim", "node_id"], how="left")


all_result = precision_recall_fscore_support(test_df["agg_label"].astype(int), test_df["flant5_agg_label"].astype(int), average="binary")

test_id_df = test_df[test_df.split=="test_ID"]
test_ood_df = test_df[test_df.split=="test_OOD"]
id_result = precision_recall_fscore_support(test_id_df["agg_label"].astype(int), test_id_df["flant5_agg_label"].astype(int), average="binary")
ood_result = precision_recall_fscore_support(test_ood_df["agg_label"].astype(int), test_ood_df["flant5_agg_label"].astype(int), average="binary")

# put results into df with precision, recall, f1
results = pd.DataFrame([all_result, id_result, ood_result], columns=["precision", "recall", "f1", "support"], index=["all", "id", "ood"])

print(results)

# test_df.to_csv(f"/net/nfs/s2-research/joeh/experiments/task3/flan_t5_{MODEL_SIZE}_predictions.csv", index=False)

