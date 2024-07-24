
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import os

metric = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


# Load the tokenizer, model, and data collator
MODEL_SIZE = "base"
MODEL_NAME = f"google/flan-t5-{MODEL_SIZE}"
OUTPUT_DIR = f"./task2_flant5_{MODEL_SIZE}/"
DATA_PATH = "joe32140/chime-sibling-coherence"

# create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# We prefix our tasks with "answer the question"
prefix = "Please answer this question: Are these sibling categories coherent with each other under the parent category? Answer 0 if any of the sibling categories are not coherent with the parent category. Answer 1 if all sibling categories are coherent with the parent category."

# Define the preprocessing function

def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    inputs = []
    for parent_category, child_categories in zip(
        examples["parent_category"], examples["child_categories"]):
        child_categories = ", ".join(f"{i+1}: {c}" for i, c in enumerate(child_categories))
        prompt = f"{prefix} Parent Category: {parent_category} Sibling Categories: {child_categories}" 
        inputs.append(prompt)
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
  
    # The "labels" are the tokenized outputs:
    labels = tokenizer(text=list(map(str, examples["agg_label"])),
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

    # flip the labels
    decoded_labels = [1 if label == 0 else 0 for label in decoded_labels]
    decoded_preds = [1 if pred == 0 else 0 for pred in decoded_preds]

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

ds = load_dataset(DATA_PATH)
df = pd.concat([
    ds["train"].to_pandas(),
    ds["val"].to_pandas(),
    ds["test_ID"].to_pandas(),
    ds["test_OOD"].to_pandas()]
)

dedup_df = df[["sibling_group_id","parent_category", "split","agg_label", "coherence_rate"]].drop_duplicates().reset_index(drop=True)
dedup_df["child_categories"] = df.groupby('sibling_group_id')["child_category"].apply(list).reset_index(drop=True)

# augment negative examples to balance the dataset
negative_df = dedup_df[(dedup_df.split=="train") & ((dedup_df.agg_label==0))].sample(128, replace=True)
dedup_df = pd.concat([dedup_df, negative_df], axis=0).reset_index(drop=True)

huggingface_dataset = Dataset.from_pandas(dedup_df)
dataset= DatasetDict({
    'train': huggingface_dataset.filter(lambda example: example['split'] == 'train'),
    'val': huggingface_dataset.filter(lambda example: example['split'] == 'val'),
    'test_ID': huggingface_dataset.filter(lambda example: example['split'] == 'test_ID'),
    'test_OOD': huggingface_dataset.filter(lambda example: example['split'] == 'test_OOD')
})

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['sibling_group_id', 'parent_category', 'child_categories', 'split'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=1e-3,
    warmup_steps=5,

    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_strategy="epoch", 
    # logging_steps=1000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="precision",
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
# Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

# flip the labels for evaluation since we are predicting the unconherent groups
decoded_labels = [1 if label == 0 else 0 for label in decoded_labels]
decoded_preds = [1 if pred == 0 else 0 for pred in decoded_preds]

test = pd.concat([dedup_df[dedup_df.split=="test_ID"], dedup_df[dedup_df.split=="test_OOD"]])
test["flan_t5_pred"] = decoded_preds
test["flan_t5_label"] = decoded_labels
test["flan_t5_pred"] = test["flan_t5_pred"].astype(int)
test["flan_t5_label"] = test["flan_t5_label"].astype(int)
test_df = pd.concat([dedup_df[dedup_df.split=="test_ID"], dedup_df[dedup_df.split=="test_OOD"]])
test_df["flan_t5_pred"] = decoded_preds
test_df["flan_t5_label"] = decoded_labels
test_df["flan_t5_pred"] = test_df["flan_t5_pred"].astype(int)
test_df["flan_t5_label"] = test_df["flan_t5_label"].astype(int)

all_result = precision_recall_fscore_support(test_df["flan_t5_label"].astype(int), test_df["flan_t5_pred"].astype(int), average="binary")

test_id_df = test_df[test_df.split=="test_ID"]
test_ood_df = test_df[test_df.split=="test_OOD"]
id_result = precision_recall_fscore_support(test_id_df["flan_t5_label"].astype(int), test_id_df["flan_t5_pred"].astype(int), average="binary")
ood_result = precision_recall_fscore_support(test_ood_df["flan_t5_label"].astype(int), test_ood_df["flan_t5_pred"].astype(int), average="binary")

# put results into df with precision, recall, f1
results = pd.DataFrame([all_result, id_result, ood_result], columns=["precision", "recall", "f1", "support"], index=["all", "id", "ood"])

test_df.to_csv(f"{OUTPUT_DIR}/flan_t5_{MODEL_SIZE}_predictions.csv", index=False)