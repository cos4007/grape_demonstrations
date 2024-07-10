# Demonstration-based Learning for Few-shot Biomedical Named Entity Recognition under Machine Reading Comprehension

## Study Description
This study reports an effective method to enhance the model’s capability to identify biomedical entities in few-shot learning. 
By redefining biomedical named entity recognition (BioNER) as a machine reading comprehension (MRC) problem, we propose a demonstration-based learning method to address few-shot BioNER, which involves constructing appropriate task demonstrations. 
In assessing our proposed method, we compared the proposed method with existing advanced methods using six datasets, including BC4CHEMD, BC5CDR-Chemical, BC5CDR-Disease, NCBI-Disease, BC2GM, and JNLPBA. 
We examined the models’ efficacy by reporting F1 scores from both the 25-shot and 50-shot learning experiments.

![Figure1](https://github.com/cos4007/grape_demonstrations/Figure1.jpg)


## Run Experiments
Follow these steps to install the required dependencies:

1. Create and activate your conda/virtual environment.

2. Run `pip install -r requirements.txt`.

3. Run `bash bert_mrc_grape_all_C.sh`.
