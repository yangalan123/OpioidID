# Opiate Addiction Identification (OpiateID)
Codebase for the NAACL 2024 Findings paper "Identifying Self-Disclosures of Use, Misuse and Addiction in Community-based Social Media Posts"

Main Authors: 

1. Chenghao Yang (yangalan1996@gmail.com) (University of Chicago, Previously at Columbia University)

2. Tuhin Chakrabarty (tuhin.chakr@cs.columbia.edu) (Columbia University)

Supervisor Team:
1. Nabila El-Bassel (School of Social Work, Columbia University)

2. Smaranda Muresan (Data Science Institute, Columbia University)


## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2023-identifying,
    title = "Identifying Self-Disclosures of Use, Misuse and Addiction in Community-based Social Media Posts",
    author = "Chenghao Yang and Tuhin Chakrabarty and Karli R Hochstatter and Melissa N Slavin and Nabila El-Bassel and Smaranda Muresan",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```

## Project Structure
1. `codebase`: Codebase for Data Processing, Fine-tuning, evaluation and visualization. 
2. `data`: The data for prompting and evaluation. As per IRB ethics approval, we kindly request the user to submit a request [here](https://docs.google.com/forms/d/e/1FAIpQLSdSkrwGQJDYwpf5ASbJKcUeDudu_hyXdlXciJiRHrdRAJ5Shg/viewform?usp=sf_link) to explain the project scope and obtained ethics approval before we send you the access to data.

## Dependency Installation
For the main repo:
```
conda create -p ./env python=3.9
conda activate ./env # the environment position is optional, you can choose whatever places you like to save dependencies. Here I choose ./env for example.
pip install -r requirements.txt
```

## Running Instructions
### Section 4: Main Experiments
1. Prompting GPT3.5/4: check out codes in `codebase/prompt_gpt4.py` for running prompts and collecting responses. Then run `post_processing_log.py` to do necessary postprocessing for normalizing the model outputs. 
2. Fine-tuning DeBERTa: check out codes in `codebase/deberta_finetuning.py`. 
### Section 5: The Role of Explanations
Check out codes in `codebase/create_sliver_rational_annotation_files.py` and `codebase/process_finetune_data_for_finetuning.py` to see how to combine gold rationale, sliver rationale and random rationale. Here we would re-use the codes in Section 4 for evaluation. 
### Section 6: Error Analysis and the Influence of Dataset Annotation Uncertainty
Check out codes in `codebase/error_analysis.py`. 