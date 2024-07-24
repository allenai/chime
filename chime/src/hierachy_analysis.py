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

    logger.info("Calculating Coverage")
    claim2path_dfs = []

    
    # Get Claude annotation
    for review_title, hier in q2hier.items():
        claim2paths_list = utils.get_claim_path_from_hierarchy(hier)
        claim2path_df = pd.DataFrame(claim2paths_list, columns=['claim_id', 'node_path'])
        claim2path_df["review_title"] = review_title
        claim2path_df["path_len"] = claim2path_df["node_path"].apply(lambda x: len(x))
        review_df = df[df["review_title"] == review_title]
        review_df["claim_id"] = np.arange(len(review_df))
        claim2path_dfs.append(
            claim2path_df.merge(review_df[["review_pmid","claim_id", "claim"]], on="claim_id")
        )
    final_claim2path_df = pd.concat(claim2path_dfs)

    final_claim2path_df["top_heading"] = final_claim2path_df["node_path"].apply(lambda x: x[0])
    final_claim2path_df["node_path_tuple"] = final_claim2path_df["node_path"].apply(lambda x: tuple(x))
    top_df = final_claim2path_df[["review_pmid", "top_heading"]].drop_duplicates()
    top_df = top_df[top_df["top_heading"] != "Uncategorized studies"]
    top_df = top_df.groupby("review_pmid").size().reset_index(name="count")

    depth_df = final_claim2path_df.sort_values(["review_pmid", "top_heading", "path_len"], ascending=False).drop_duplicates(["review_pmid", "top_heading"])
    depth_df = depth_df[depth_df["top_heading"] != "Uncategorized studies"]
    print("Hier Depth Count")
    print(depth_df.path_len.describe().to_frame())
    
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

    # sibling count
    exploded_sibling_df = hierarchy_annotation.explode("sibling_categories")
    exploded_sibling_df = exploded_sibling_df.dropna(subset="sibling_categories")
    exploded_sibling_df["sibling_count"] = exploded_sibling_df["sibling_categories"].apply(lambda x: len(x[-1]))
    print(exploded_sibling_df["sibling_count"].describe().to_frame())

    # coverage
    hierarchy_annotation["claim_used"] = hierarchy_annotation.claim_ids.apply(len)
    print(hierarchy_annotation[hierarchy_annotation.aspect == "Uncategorized studies"].claim_used.describe().to_frame())
    
    # claim / hierachy
    print((hierarchy_annotation[hierarchy_annotation.aspect != "Uncategorized studies"]["claim_used"] / hierarchy_annotation[hierarchy_annotation.aspect != "Uncategorized studies"]["review_claim_count"]).describe().to_frame())

    # uncategorized claim count
    print(hierarchy_annotation[hierarchy_annotation.aspect == "Uncategorized studies"].claim_used.describe().to_frame())
