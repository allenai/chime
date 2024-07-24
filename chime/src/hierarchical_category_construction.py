import pandas as pd
import lm_api
import os
import spacy
from prompt_library import FULL_CLAIM_ASPECT_PROMPT, FINDING_EXTRACTION_PROMPT

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CACHE_DIR = ".cache/lm_cache/"
OUTPUT_DIR = ".cache/hierarchy_outputs/"
DEBUG = True

MODEL = "gpt-3.5-turbo-0613"
openai_model = lm_api.LanguageModelAPI(
    "openai",
    MODEL,
    CACHE_DIR,
    OPENAI_API_KEY)

claude_model = lm_api.LanguageModelAPI(
    "claude",
    "claude-2",
    CACHE_DIR,
    CLAUDE_API_KEY)

nlp = spacy.load("en_core_sci_sm")

def finding_extraction(df_sampled):
    def extract_findings(title, text):
        input_text = f"Title:\n{title}\n\nAbstract:\n{text}\n\n{FINDING_EXTRACTION_PROMPT}"
        return openai_model.chat(input_text, max_tokens=128)
    print(extract_findings(df_sampled["study_title"].iloc[0], df_sampled["study_abstract"].iloc[0]))
    
    df_sampled["claim"] = df_sampled.apply(lambda row: extract_findings(row.study_title, row.study_abstract), axis=1)
    return df_sampled

def etract_entities_from_study_abstract(df):
    # use pipe to speed up
    abstracts = df["study_abstract"].values.tolist()
    study_entities = []
    for doc in nlp.pipe(abstracts, batch_size=256, n_process=64):
        study_entities.append([e.text.lower() for e in doc.ents])
    df["study_entity"] = study_entities
    
    
    # %%
    from collections import Counter
    grouped_data = df.groupby('review_pmid')['study_entity'].apply(list).reset_index(name='word_list')
    def count_words(word_list):
        expanded_word_list = [item for sublist in word_list for item in set(sublist)]
        word_count = Counter(expanded_word_list)
        return word_count.most_common()
    grouped_data['word_count'] = grouped_data['word_list'].apply(count_words)
    grouped_data["frequent_entity"] = grouped_data["word_count"].apply(lambda x: [e[0] for e in x[:20]])
    df = df.merge(grouped_data[["review_pmid", "frequent_entity"]], on="review_pmid")
    return df

def generate_aspect_hierarchy(df):
    claim_aspects = []
    print("Generating aspect hierarchy")
    for review_pmid in df.review_pmid.unique():
        print(f"\r{review_pmid}", end=" ")
        entities = ", ".join([f"{e}" for i, e in enumerate(df[df.review_pmid==review_pmid].iloc[0].frequent_entity )])
        claims = "".join([f"Claim {i}: {c}\n" for i, c in enumerate(df[df.review_pmid==review_pmid].claim )])
        review_title = df[df.review_pmid==review_pmid].review_title.iloc[0]
        input_text = f"**Review itle**\n{review_title}\\Frequent entities from study abstracts:\n{entities}\n\n**Study Claim List**\n{claims}\n\n{FULL_CLAIM_ASPECT_PROMPT}"
        claude_aspect = claude_model.chat(input_text, max_tokens=2048)
        
        claim_aspects.append({"review_pmid": review_pmid, "claude_claim_hierarchy": claude_aspect})
        
    claim_aspect_df = pd.DataFrame(claim_aspects)
    df = df.merge(claim_aspect_df, on="review_pmid")
    return df


if __name__ == "__main__":

    # check if cache dir exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv("../../resources/raw_source_data.csv")
    if DEBUG:
        print("========= DEBUG MODE ===========")
        df = df[df.review_pmid.isin([25720328, 22895927])]
    
    print("Number of reviews:", len(df.review_pmid.unique()))
    
    print("Claim extraction")
    df = finding_extraction(df)
    
    print("Extract abstract enitities")
    df = etract_entities_from_study_abstract(df)
    
    print("Generate aspect hierarchy")
    df = generate_aspect_hierarchy(df)
    df.to_csv(f"{OUTPUT_DIR}processed_data.csv", index=False)



