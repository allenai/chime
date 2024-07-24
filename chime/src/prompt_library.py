FINDING_EXTRACTION_PROMPT = """
Task:
Conclude new findings and null findings from the abstract in one sentence in the atomic format. Do not separate new findings and null findings. The finding must be relevant to the title. Do not include any other information.

Definition:
A scientific claim is an atomic verifiable statement expressing a finding about one aspect of a scientific entity or process, which can be verified from a single source.
"""

FULL_CLAIM_ASPECT_PROMPT = """
**Instruction**
Your task is to process a review title involving relevant clinical studies as per the following requirements:

1. **Top-Level Aspect Generation:** Utilize the entities extracted from the study abstracts for identifying up to 5 top-level aspects from the clinical study claims. You should list these aspects in a bulleted list format without incorporating any extraneous information. Cite the entities in that support the aspects. This will be the [Response 1] section.

2. **Hierarchical Faceted Category Generation:** For every top-level aspect in [Response 1], proceed to generate hierarchical faceted categories that closely align with the above study claims. The granularity of these categories must be similar to their corresponding parent categories and the siblings categories. Avoid including unrelated information. Cite the claims that support your categories. This will make up the [Response 2] section of your output.

**Remember**
1. Precision is vital in this process; strive to avoid vague or imprecise extractions.
2. Include only relevant data and exclude any information not pertinent to the task.
3. Strictly adhere to the output format. The claims are cited in the format "(Claim 0, 2, 3, 12)" for each category and aspect.
4. The output should be in the form of a nested list using numbers. 


Here is an example:

If given the review title "The efficacy of Remdesivir in treating COVID-19 patients: A review," your task output might look like this:

Frequent entities from study abstracts:
Efficacy, Remdesivir, treatment, COVID-19 patients

**Output Format**
[Response 1]:
Aspect 1: Efficacy of treatment (Efficacy)
Aspect 2: Application of Remdesivir (Remdesivir)
Aspect 3: Treatment of COVID-19 patients (treatment, COVID-19 patients)

[Response 2]:
Aspect 1: Efficacy of treatment (Claim 0, 2, 3, 12)
    1: Efficiency of alternative treatments (Claim 0, 2, 3, 12)
        1.1: Efficacy of Remdesivir (Claim 0, 12)
        1.2: Efficacy of other drugs (Claim 3)
    2: Side-effects comparison (Claim 2)
Aspect 2: Application of Remdesivir (Claim 2, 4, 5, 6, 7, 8, 9, 10, 11)
    1: Usage of other drugs (Claim 4, 5, 6, 9, 10, 11)
    2: Dosing comparisons (Claim 7, 8)
        2.1: Dosing of Remdesivir (Claim 7)
Aspect 3: Treatment of COVID-19 patients (Claim 1, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    1: Treatment procedures for other diseases (Claim 13, 14, 16, 17, 18, 19, 20, 21)
    2: Treatment timeframe comparisons. (Claim 1, 15)
"""


TASK2_CoT_GENERATION_PROMPT = """
** Instruction **
In this task, you will be annotating the relationship among a set of sibling categories.You will assess whether these sibling categories logically belong together within their shared parent category, a concept referred to as 'coherence'. 


Your task is to label whether ALL sibling categories are coherent with each other. 
If all sibling categories fit well and logically belongs to the broader group, label it 'These sibling categories are coherent' to signify its coherence. Make sure silbings are at the same level of granularity for coherence assessment.
If any category doesn't seem to belong logically or doesn't fit well within the group, label it 'These sibling categories are NOT coherent' to indicate non-coherence.

Your decisions should be based solely on the level of coherence – how well these categories fit together under their shared parent category and not on any other factors or personal preferences.

**Remember**
1. You should start with step-by-step reasoning and generate the answer at the end in the given format.
2. You should only reply with the answer in the format of [These sibling categories are coherent] or [These sibling categories are NOT coherent].
3. You will be given a parent category and a set of sibling categories. You should assess each sibling category independently.

Again, follow the format below to reply:

Step-by-step reasoning:

[Your reasoning]

Answer:
[These sibling categories are coherent] or [These sibling categories are NOT coherent]

** Question **
Parent category: {parent_category}

Sibling categories: {sibling_categories}
"""

TASK3_CoT_GENERATION_PROMPT = """
** Instruction **
In this task, your role as an annotator is to assess whether a specific claim belongs to a provided category.

Your responsibility is to assign a binary label for each category-claim pairing:
1. "The claim belongs to the category" -  Choose this if any part or aspect of the claim is relevant to the category, even if the connection is broad or indirect. This includes claims that are negations or opposites of the category. See the following examples:
The claim “Assisted hatching through partial zona dissection does not improve pregnancy and embryo implantation rates in unselected patients undergoing IVF or ICSI” belongs to “Impact on specific patient groups” category because patient groups can be applied to not only patient demographics but also patients with the same disease/symptom.
The claim “Sumatriptan is effective in reducing productivity loss due to migraine, with significant improvements in productivity loss and return to normal work performance compared to placebo.” belongs to “Headache relief” because headache is one of the symptoms of migraine even though it is not explicitly mentioned in the claim.

2. "The claim does NOT belong to the category" - Choose this if there is no meaningful connection between the claim and the category.


**Remember**
1. Only reply with the answer in the format of [The claim belongs to the category] or [The claim does NOT belong to the category].
2. Do not reply with any other format.
3. Start with step-by-step reasoning and generate the answer at the end in the given format.

**Claim**
{}

**Category**
{}

Again, follow the format below to reply:

Step-by-step reasoning:
[Your reasoning]

Answer:
[The claim belongs to the category] or [The claim does NOT belong to the category]
"""
