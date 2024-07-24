# CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for Literature Review Support

This is the repository of the implementation in the [paper](https://arxiv.org/abs/2407.16148).

## Environment Setup

Conda environment is recommended for running the code. To create the environment, run the following command:
```bash
conda env create -f environment.yml
```

Download SciSpacy model:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

To activate the environment, run the following command:
```bash
conda activate chime
```

Add api keys to the environment variable:
```bash
export CLAUDE_API_KEY="YOUR KEY"
export OPENAI_API_KEY="YOUR KEY"
```

## Hierearchy Generation Pipeline

THe hierarchy generation pipeline is implemented in `chime/src/hierarchical_category_construction.py`. 

> Turn off DEBUG flag in `chime/src/hierarchical_category_construction.py` to run the pipeline on the entire dataset.

```bash
cd chime/src
python hierarchical_category_construction.py
```
## Fine-tuning and LLM Prediction

The fine-tuning and LLM prediction is implemented in `chime/src/flanT5` and `chime/src/llm_prediction` respectively.

## Released datasets and the fine-tuned model

Finetuned model and dataset are available on theHugging Face hub. 
You can find the datasets as follows:
1. `joe32140/chime-parent-child-relation` for the parent-child relation dataset: [link](https://huggingface.co/datasets/joe32140/chime-parent-child-relation).
2. `joe32140/chime-sibling-coherence` for the sibling coherence dataset: [link](https://huggingface.co/datasets/joe32140/chime-sibling-coherence).
3. `joe32140/chime-claim-category` for the claim and category relevance dataset: [link](https://huggingface.co/datasets/joe32140/chime-claim-category).

The finetuned model for claim and category relevance prediction `joe32140/flan-t5-large-claim-category`: [link](https://huggingface.co/joe32140/flan-t5-large-claim-category).

We also provide the model prediction on hierachy without human annotion for the claim and category prediction. You can find the prediction [here](https://huggingface.co/joe32140/flan-t5-large-claim-category).

## Parse the generated hierarchy
See `chime/src/parse_generated_hierarchy.py` parse the generated hierarchy into structured format. Note that 2 out of 474 cannot be parsed due to the format of the generated hierarchy which results in total of 472 hierarchies in the paper.

## Additional Resources

- [Paper](https://arxiv.org/abs/2407.16148)
- `resources/raw_generated_hierarchy.csv` contains the raw generated hierarchies from the Claude-2.
- `resources/raw_source_data.csv` contains the raw review and studies data from the Cochrane Library. The generated claims are also included in this file.

## Citation

If you use this code or dataset, please cite the following:

```
@article{hsu2024chime,
  title = {CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for Literature Review Support},
  author = {Hsu, Chao-Chun and Bransom, Erin and Sparks, Jenna and Kuehl, Bailey and Tan, Chenhao and Wadden, David and Wang, Lucy Lu and Naik, Aakanksha},
  year = {2024},
  month = aug,
  journal = {ACL Findings},
}
```
