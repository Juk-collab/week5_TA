# Week5 Technical assignment

## General 

This task was about modifying and running a given .ipynb file in order to finetune a small LLM with PEFT and evaluting the results.

TinyLlama/TinyLlama-1.1B-Chat-v1.0 model was chosen and finetuned with PEFT, the finetuning seemed to work fine based on both Qualitative and Quantitative testing.

Snapshots folder contains snapshots mostly from the results of Qualitative and Quantitative testing, and huggingface_model.png is a snapshot of the finetuned model uploaded to hugging face.

Snapshots/abs_improvement.png show clear improvement on ROUGE metrics on an evaluation dataset after the model is finetuned.

Snapshots/Summaries.png, as well as Snapshots/pretrained_output_example.png and Snapshots/finetuned_output_example.png show examples of the output of the model before and after fine-tuning with PEFT. One can see the model performance w.r.t summarizing the text improved qualitatively after finetuning. While the finetuned model's examples are still not entirely accurate, they are a lot better than the outputs of the pretrained model. 

Snapshots/training.png shows the training and validation loss during the finetuning, which shows the losses have reduced as one would expect.

All the snapshots except the huggingface_model.png are simply taken from the outputs of by week5-technical.ipynb

Link to this git repo: https://github.com/Juk-collab/week5_TA


## Errors Encountered and Fixes

While setting up the project, several potential errors were encountered, primarily during the installation of dependencies. Below are the details of the issues and their resolutions:

## Dependency Conflicts

The following errors were identified during installation:

- **gensim 4.3.3** requires `scipy<1.14.0,>=1.7.0`, but the current version is `scipy 1.15.1`, which causes a compatibility issue.
- **mlxtend 0.23.3** requires `scikit-learn>=1.3.1`, but the current version is `scikit-learn 1.2.2`, resulting in an incompatibility.
- **plotnine 0.14.4** requires `matplotlib>=3.8.0`, but the current version is `matplotlib 3.7.5`, which is incompatible.

### Solution:
To fix these issues, I installed compatible versions of the libraries:

```bash
pip install scipy==1.13.0
pip install scikit-learn==1.3.1
pip install matplotlib==3.8
```

These versions are compatible with the libraries mentioned above and resolved the dependency conflicts.

## Rouge-Score Package Installation

The `rouge_score` package name has changed. To install it, used the following command:

```bash
pip install rouge-score
```

Note that although both `rouge_score` and `rouge-score` work for installation via pip, the correct name is `rouge-score`. However, the package is still imported as `rouge_score` in the code.

Additionally, it seems that the `rouge_score` package does not have version information, so .__version__ was omitted from version printing for rogue_score.

## Other Adjustments

No major errors were encountered, though there were a few adjustments made to paths and configuration based on my own setup. These changes are expected, as running the code in a personal environment requires these tweaks.

### Path Adjustments:
- The checkpoint path needed to be updated from `checkpoint-1000` to `checkpoint-500` in the following line of code, as the instructions specify running for 500 iterations:

```python
ft_model = PeftModel.from_pretrained(base_model, "/kaggle/working/peft-dialogue-summary-training/final-checkpoint/checkpoint-500", torch_dtype=torch.float16, is_trainable=False)
```

This change ensures that the correct finetuned model checkpoint is used for the evaluation.
