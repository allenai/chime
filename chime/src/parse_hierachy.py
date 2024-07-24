import pandas as pd
import sys
sys.path.append('/home/joeh/hierarchy-organization/my_project/src')
import utils
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def print_tree(data, level=0):
    output = []
    name, _ ,sub_cats = data
    output.append('\t'*level + '- ' + name)
    
    for item in sub_cats:
        output.extend(print_tree(item, level+1))
    return output

def load_data(df):
    q2papers ={}
    q2hier = {}
    pmid2q = {}
    id2q = {}
    
    for i, review_pmid in enumerate(df.review_pmid.unique()):
        tmp_df = df[df.review_pmid == review_pmid]
        title = tmp_df.review_title.iloc[0]
        pmid2q[review_pmid] = title
        id2q[i+1] = title
        
    
    for review_pmid in df.review_pmid.unique():
        papers = []
        tmp_df = df[df.review_pmid == review_pmid]
        for i, row in tmp_df.iterrows():
            papers.append({'study_title': row.study_title, 'study_abstract': row.study_abstract, 'claim':row.claim})
        q2papers[tmp_df.review_title.iloc[0]] = papers
        try:
            q2hier[tmp_df.review_title.iloc[0]] = utils.get_hierarchy(row.claude_claim_hierarchy)
        except:
            print("==== Fail ====")
            # print(review_pmid)
            # import sys
            # sys.exit()
    coverage = []
    for review_title, hier in q2hier.items():
        papers = q2papers[review_title]
        claim_full_set = set(list(range(len(papers))))
        categorized_claim_set = utils.get_claim_set_from_hierarchy(hier)

        uncategorized_claim_set = claim_full_set - categorized_claim_set
        coverage.append(len(categorized_claim_set) / len(claim_full_set))
        # print(uncategorized_claim_set)
        if len(uncategorized_claim_set) > 0:
            hier.append(["Uncategorized studies", sorted(list(uncategorized_claim_set)), []])
    # print(hier)
    return q2papers, q2hier, pmid2q, id2q, coverage



if __name__ == "__main__":

    raw_df = pd.read_csv("../../resources/raw_source_data.csv")
    hier_df = pd.read_csv("../../resources/raw_generated_hierarchy.csv")
    df = raw_df.merge(hier_df[["review_id", "claude_claim_hierarchy"]], on="review_id")

    review_pmid_df = df[["review_title", "review_pmid"]].drop_duplicates()
    
    logger.info("Number of reviews: {}".format(review_pmid_df.shape[0]))
    
    logger.info("Loading Hierarchies")
    q2papers, q2hier, pmid2q, id2q, coverage = load_data(df)
    
    review2hier = []

    for review_title, hier in q2hier.items():
        for aspect in hier:
            
            claim_ids = list(utils.get_claim_set_from_hierarchy([aspect]))
            categories = utils.get_category_list_from_hierarchy([aspect])
            category_paths = utils.get_category_paths_from_hierarchy([aspect])
            categories_pairs = utils.get_category_pairs_from_hierarchy([aspect], is_root=True)
            sibling_categories = utils.get_sibling_categories_with_parent_from_hierarchy([aspect], is_root=True)
            aspect_name, claims, sub_cats = aspect
            tree_string = print_tree(aspect)
            joined_string = "\n".join(tree_string)
            review2hier.append([review_title, aspect_name, aspect, joined_string, categories, category_paths, categories_pairs, sibling_categories, claim_ids])
            
    hierarchy_annotation = pd.DataFrame(review2hier, columns = ["review_title", "aspect", "aspect_hier", "hierarchy", "categories", "category_paths", "categories_pairs", "sibling_categories", "claim_ids"])
    hierarchy_annotation = hierarchy_annotation.merge(review_pmid_df, on="review_title")
    hierarchy_annotation["hierarchy_id"] = hierarchy_annotation.groupby("review_pmid").cumcount()
    claim_count_df = df.groupby("review_pmid")["review_pmid"].count().rename("review_claim_count").reset_index()
    hierarchy_annotation = hierarchy_annotation.merge(claim_count_df, on="review_pmid")
    
    print(hierarchy_annotation.columns)