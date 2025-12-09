# Commonsense Validation and Explanation (ComVE) Project

This repository contains implementations for all three subtasks of the [ComVE](https://competitions.codalab.org/competitions/21080) shared task from SemEval-2020. The project focuses on evaluating whether systems can determine if a natural language statement makes sense and provide explanations.

## Project Overview

### SubTask A – Statement Pair Classification

* **Task:** Given two statements, classify which one does not make sense.
* **Approach:** Fine-tuned **RoBERTa** for binary classification using the Transformers `Trainer`.
* **Evaluation:** Accuracy.

### SubTask B – Reason Selection

* **Task:** Given a nonsensical statement and three candidate reasons, select the correct reason.
* **Approach:** Treated as a multiple-choice problem; fine-tuned **RoBERTa** using a `Trainer` with a `DataCollatorForMultipleChoice`.
* **Evaluation:** Accuracy.

### SubTask C – Reason Generation

* **Task:** Generate a valid reason explaining why a statement does not make sense.
* **Approach:** Treated as a sequence-to-sequence problem; fine-tuned **BART** using `Seq2SeqTrainer`.
* **Evaluation:** BLEU and ROUGE scores.

## Dataset

* **Training:** 10,000 statements for SubTask C; 9,997 statement pairs for SubTask A; 9,997 statements for SubTask B
* **Development:** 997 examples
* **Test:** 1,000 examples

All datasets were pre-processed using the Hugging Face `Datasets` library and tokenized using the model-specific tokenizer.

## Requirements

* Python 3.9+
* `transformers` >= 4.27.2
* `datasets` >= 2.10.0
* `evaluate` >= 0.4.0
* `torch` >= 2.1.0
* `numpy`
* `pandas`

Optional for GPU training:

* CUDA-enabled GPU or TPU (Google Colab recommended for full-scale training)

## Usage

1. Load the dataset using provided functions.
2. Load the pre-trained model and tokenizer (`RoBERTa` for SubTask A/B, `BART` for SubTask C).
3. Pre-process and tokenize the dataset according to the subtask requirements.
4. Fine-tune the model using `Trainer` or `Seq2SeqTrainer`.
5. Make predictions on the test set.
6. Evaluate performance using accuracy (SubTask A/B) or BLEU/ROUGE (SubTask C).

## Notes

* Reduced datasets and base models were used for quicker experimentation (`shrink_dataset=True`, `base_model=True`).
* Full training with the entire dataset and large models can be performed in Google Colab.
* Random seed initialization ensured reproducibility, though minor variations may occur due to hardware/software differences.
