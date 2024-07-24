import re
import pickle
import os 
import pandas as pd
import numpy as np

def clean_claim(claim):
    return claim.replace("New finding: ", "").replace("Null finding: ", "").replace("New findings: ", "").replace("Null findings: ", "")

def clean_aspect(aspect):
    pattern = r'\([^()]*\)'
    return re.sub(pattern, '', aspect).strip()

def extract_response(full_text):
    start = full_text.find("[Response 2]")
    end = full_text.find("Review:", start)
    if end == -1:  # If there is no "Review:" after "[Response 2]", get the rest of the text
        end = full_text.find("Human:", start)
        if end == -1:
            end = len(full_text)
    return full_text[start+13:end].strip()

def get_aspect_hierarchies(response):
    lines = response.split('\n')
    output = {}
    aspect2claims = {}
    flag = 0
    aspect= None
    tmp = []
    list_type = None
    sub_list_type = None
    
    def get_next_non_empty_sent(lines, i):
        while i < len(lines):
            if lines[i].strip() != "":
                return lines[i]
            i += 1
        return lines[i]
    
    # detect type by first line
    if lines[0].startswith("Aspect"):
        list_type = 1 # Aspect: 1.1.1
    elif lines[0].startswith("- Aspect"):
        list_type = 4
    elif lines[0].startswith("- "):
        list_type = 2 # - Aspect
    elif lines[0][0].isnumeric():
        list_type = 5 # Aspect -
    elif lines[0][0].isalnum():
        list_type = 3 # Aspect -
        
        
    next_sent =  get_next_non_empty_sent(lines, 1)
    if next_sent.strip().startswith("1"):
        sub_list_type = 1
    elif next_sent.strip().startswith("-"):
        if list_type == 4:
            sub_list_type = 3
        else:
            sub_list_type = 2
    
    
    for line in lines:
        if line.strip().startswith("Aspect") and list_type == 1:
            if flag == 1:
                output[aspect] = tmp
                tmp = []
            s, claims = parse_claims(line)
            
            flag = 1
            try:
                aspect = s.split(":")[1].strip()
                aspect = clean_aspect(aspect)
            except:
                aspect = s.split("-")[1].strip()
                aspect = clean_aspect(aspect)
            aspect2claims[aspect] = claims
            tmp = []
        elif line.strip().startswith("- Aspect") and list_type == 4:
            if flag == 1:
                output[aspect] = tmp
                tmp = []
            s, claims = parse_claims(line)
            
            flag = 1
            try:
                aspect = s.split(":")[1].strip()
                aspect = clean_aspect(aspect)
            except:
                aspect = s.split("-")[1].strip()
                aspect = clean_aspect(aspect)
            aspect2claims[aspect] = claims
            tmp = []
        elif line.startswith("- ") and list_type == 2:
            if flag == 1:
                output[aspect] = tmp
                tmp = []
            s, claims = parse_claims(line)
            
            flag = 1
            aspect = s.replace("- ", "").strip()
            aspect = clean_aspect(aspect)
            aspect2claims[aspect] = claims
            tmp = []
        elif line and ((not line[0] in [" ","-"]) and re.match(r'^(\d\.)+(\d*)', line.strip()) is None)  and list_type == 3:
            if flag == 1:
                output[aspect] = tmp
                tmp = []
            s, claims = parse_claims(line)
            flag = 1
            aspect = s.strip()
            aspect = clean_aspect(aspect)
            aspect2claims[aspect] = claims
            tmp = []
        elif line and (line[0].isnumeric())  and list_type == 5:
            if flag == 1:
                output[aspect] = tmp
                tmp = []
            s, claims = parse_claims(line)
            flag = 1
            aspect = s.strip()
            aspect = clean_aspect(aspect)
            aspect2claims[aspect] = claims
            tmp = []
        else:
            if line.strip() == "":
                continue
            tmp.append(line)
    if flag == 1:
        output[aspect] = tmp
    return output, aspect2claims, list_type, sub_list_type
        
def parse_claims(s):
    original_s = s
    s = s.replace("(Claims", "(Claim")
    s = s.replace("(Claim:", "(Claim")
    claims = re.findall(r'Claim (\d+(?:, \d+)*)', s)
    claims = [int(c) for c in re.split(', ', claims[0])] if claims else []
    s = re.sub(r' \(Claim .*\)', '', s)
    
    
    if len(claims) == 0:
        s = original_s.replace("(Papers", "(Paper")
        s = s.replace("(Paper:", "(Paper")
        claims = re.findall(r'Paper (\d+(?:, \d+)*)', s)
        claims = [int(c) for c in re.split(', ', claims[0])] if claims else []
        s = re.sub(r' \(Paper .*\)', '', s)
    return s, claims


def parse_list(lst, level=0):
    if level >= len(lst):
        return {}
    
    s, claims = parse_claims(lst[level])
    key, rest = s.split('.', 1)
    rest = rest.strip()
    if rest:
        return {key: [rest, claims, parse_list(lst, level+1)]}
    else:
        return {key: [claims, parse_list(lst, level+1)]}


def to_nested_list_type_one(lst, current_row, current_level):
    """
    1
    1.1
    """
    nested_list = []
    while current_row < len(lst):
        if lst[current_row].strip() == "":
            current_row += 1
            continue
        row = lst[current_row].strip()
        s, claims = parse_claims(row)
        # print(s, claims)
        level = s.split(' ', 1)[0].strip(".")
        level = len(level.split('.')) 
        title = s.split(' ', 1)[1]
        if level == current_level:
            nested_list.append([title, claims, []])
            current_row += 1
        elif level < current_level:
            return nested_list, current_row
        else:
            ret_list, current_row = to_nested_list_type_one(lst, current_row, current_level+1)
            if len(ret_list) != 0:
                nested_list[-1][-1].extend(ret_list)
        assert type(current_row) == int, "current_row is not int"
    return nested_list, current_row

def to_nested_list_type_two(lst, current_row, current_level):
    """
    - 
      -
    """
    nested_list = []
    while current_row < len(lst):
        if lst[current_row].strip() == "":
            current_row += 1
            continue
        row = lst[current_row]
        s, claims = parse_claims(row)
        level = s.split('-', 1)[0]
        level = len(level) 
        try:
            title = s.split('-', 1)[1].strip()
        except Exception as e:
            return e
            print(s)
            import sys
            sys.exit()
        if level == current_level:
            nested_list.append([title, claims, []])
            current_row += 1
        elif level < current_level:
            return nested_list, current_row
        else:
            ret_list, current_row = to_nested_list_type_two(lst, current_row, level)
            if len(ret_list) != 0:
                nested_list[-1][-1].extend(ret_list)
        assert type(current_row) == int, "current_row is not int"
        
    return nested_list, current_row


def to_nested_list_type_three(lst, current_row, current_level):
    """
    - 1
      - 1.1
    """
    nested_list = []
    while current_row < len(lst):
        if lst[current_row].strip() == "":
            current_row += 1
            continue
        row = lst[current_row]
        s, claims = parse_claims(row)
        level = s.split('-', 1)[0]
        level = len(level) 
        try:
            title = s.strip().replace("- ", "").split(' ', 1)[1].strip()
        except Exception as e:
            return e
            print(s)
            import sys
            sys.exit()
        if level == current_level:
            nested_list.append([title, claims, []])
            current_row += 1
        elif level < current_level:
            return nested_list, current_row
        else:
            ret_list, current_row = to_nested_list_type_two(lst, current_row, level)
            if len(ret_list) != 0:
                nested_list[-1][-1].extend(ret_list)
        assert type(current_row) == int, "current_row is not int"
        
    return nested_list, current_row


def to_nested_list(lst, current_row, current_level, list_type):
    if list_type == 1:
        return to_nested_list_type_one(lst, current_row, current_level)
    elif list_type == 2:
        return to_nested_list_type_two(lst, current_row, current_level)
    elif list_type == 3:
        return to_nested_list_type_three(lst, current_row, current_level)
    assert False, "list_type is not valid"
        

def get_hierarchy(hier):
    response = extract_response(hier)
    # print(response)
    aspect_dic, aspect2claim, list_type, sub_list_type = get_aspect_hierarchies(response)
    # print(list_type, sub_list_type, aspect_dic)
    output = []
    for aspect, response in aspect_dic.items():
        if len(response) != 0:
            if sub_list_type == 1:
                s, claims = parse_claims(response[0])
                baseline_level = s.split(' ', 1)[0].strip(".")
                baseline_level = len(baseline_level.split('.'))
                # print("baseline level", aspect, baseline_level)
            elif sub_list_type in [2, 3]:
                baseline_level = len(response[0].split('-', 1)[0])
            else:
                baseline_level = 0
                
            nested_list, _ = to_nested_list(response, 0, baseline_level, sub_list_type)
            output.append([aspect, aspect2claim[aspect], nested_list])
        else:
            output.append([aspect, aspect2claim[aspect], []])
    
    return output
        
def get_claim_set_from_hierarchy(hierarchy):
    
    unique_claims = set()
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        unique_claims.update(claims)
        unique_claims.update(get_claim_set_from_hierarchy(sub_nodes))
    return unique_claims

def get_claim_category_pair_from_hierarchy(hierarchy):
    
    claim_category_pairs = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        claim_category_pairs.extend([(c, aspect) for c in claims])
        claim_category_pairs.extend(get_claim_category_pair_from_hierarchy(sub_nodes))
    return claim_category_pairs

def get_category_list_from_hierarchy(hierarchy):
    
    unique_categories = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        unique_categories.extend([aspect])
        unique_categories.extend(get_category_list_from_hierarchy(sub_nodes))
    return unique_categories

def get_category_paths_from_hierarchy(hierarchy, path=[]):
    unique_categories = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        unique_categories.extend([path+[aspect]])
        unique_categories.extend(get_category_paths_from_hierarchy(sub_nodes, path+[aspect]))
    return unique_categories

def get_claim_path_from_hierarchy(hierarchy, path=[]):
    unique_claims = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        new_path = path + [aspect]
        unique_claims.extend([(c, new_path) for c in claims])
        unique_claims.extend(get_claim_path_from_hierarchy(sub_nodes, new_path))
    return unique_claims

def get_node_path_from_hierarchy(hierarchy, path=[], seen=[]):
    unique_categories = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        node_id = len(seen)
        seen.append(node_id)
        unique_categories.extend([path+[node_id]])
        unique_categories.extend(get_node_path_from_hierarchy(sub_nodes, path+[node_id], seen))
    return unique_categories

def get_claim_leaf_path_from_hierarchy(hierarchy, path=[]):
    unique_claims = []
    for node in hierarchy:
        aspect, claims, sub_nodes = node
        new_path = path + [aspect]
        if len(sub_nodes) == 0:
            unique_claims.extend([(c, new_path) for c in claims])
        unique_claims.extend(get_claim_leaf_path_from_hierarchy(sub_nodes, new_path))
    return unique_claims

def get_category_pairs_from_hierarchy(hierarchy, is_root=False):
    category_pairs = []
    silbing_categories = []
    for node in hierarchy:
        category, _, sub_nodes = node
        silbing_categories.append(category)
        sub_categories, category_pairs_from_child = get_category_pairs_from_hierarchy(sub_nodes)
        for sub_category in sub_categories:
            category_pairs.append((category, sub_category))
        category_pairs = category_pairs + category_pairs_from_child
    if is_root:
        return category_pairs
    else:
        return silbing_categories, category_pairs


def get_sibling_categories_from_hierarchy(hierarchy, is_root=False):
    if len(hierarchy) == 0:
        return 
    sibling_categories = []
    current_sibling_categories = []
    for node in hierarchy:
        category, _, sub_nodes = node
        sub_sibling_sets = get_sibling_categories_from_hierarchy(sub_nodes)
        if sub_sibling_sets:
            sibling_categories.extend(sub_sibling_sets)
        current_sibling_categories.append(category)
    
    if is_root:
        return sibling_categories 
    else:
        sibling_categories.append(current_sibling_categories)
        return sibling_categories
    
def get_sibling_categories_with_parent_from_hierarchy(hierarchy, parent_category=None, is_root=False):
    if len(hierarchy) == 0:
        return 
    sibling_categories = []
    current_sibling_categories = []
    for node in hierarchy:
        category, _, sub_nodes = node
        sub_sibling_sets = get_sibling_categories_with_parent_from_hierarchy(sub_nodes, category)
        if sub_sibling_sets:
            sibling_categories.extend(sub_sibling_sets)
        current_sibling_categories.append(category)
    
    if is_root:
        return sibling_categories
    else:
        sibling_categories.append((parent_category, current_sibling_categories))
        return sibling_categories


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
            papers.append({'title': row.study_title, 'abstract': row.study_abstract, 'claim':row.claim})
        q2papers[tmp_df.review_title.iloc[0]] = papers
        try:
            q2hier[tmp_df.review_title.iloc[0]] = get_hierarchy(row.claude_claim_hierarchy)
        except:
            print("==== Fail ====")
            # print(row.claude_claim_hierarchy)
            import sys
            sys.exit()
    
    for review_title, hier in q2hier.items():
        papers = q2papers[review_title]
        claim_full_set = set(list(range(len(papers))))
        categorized_claim_set = get_claim_set_from_hierarchy(hier)
        # get the set of claims that are not categorized
        uncategorized_claim_set = claim_full_set - categorized_claim_set
        if len(uncategorized_claim_set) > 0:
            hier.append(["Uncategorized studies", sorted(list(uncategorized_claim_set)), []])
    # print(hier)
    return q2papers, q2hier, pmid2q, id2q
