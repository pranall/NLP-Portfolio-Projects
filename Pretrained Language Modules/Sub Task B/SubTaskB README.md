# Pre-trained Language Models: SubTask B

This repository contains the implementation and fine-tuning of a Pre-trained Language Model for **SubTask B** of the [ComVE](https://competitions.codalab.org/competitions/21080) shared task from SemEval-2020. The goal of SubTask B was to evaluate whether a system can identify the reason a given nonsensical statement violates common sense.

## Task Description

For SubTask B, the input consisted of a nonsensical statement and three possible explanations. The model needed to predict which explanation correctly describes why the statement is against common sense. Example:

**Statement:** He put an elephant into the fridge.
**Reason A:** An elephant is much bigger than a fridge.
**Reason B:** Elephants are usually white while fridges are usually white.
**Reason C:** An elephant cannot eat a fridge.

The correct label in this example is **Reason A**. This problem was treated as a Multiple Choice task, where the input was the statement paired with each explanation and the output was the index of the correct reason.

## Dataset

The **ComVE** dataset for SubTask B consisted of:

* **Train set:** 9997 nonsensical statements with 3 possible reasons each
* **Development set:** 997 statements
* **Test set:** 1000 statements

Labels were encoded as `0`, `1`, or `2` corresponding to **Reason A**, **B**, and **C**.

## Model

The model was fine-tuned using the [Transformers](https://huggingface.co/docs/transformers/index) library. In this project, [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) was used due to its pre-training on large corpora and suitability for downstream NLP tasks.

## Implementation Steps

1. **Loading the Model and Tokenizer:**
   The `load_model` function loaded the pre-trained RoBERTa model and its tokenizer using `AutoModelForMultipleChoice`.

2. **Data Preprocessing:**
   The dataset was tokenized using the tokenizer. Each statement was paired with all three reasons, producing input sequences that were padded and truncated to a fixed `max_length`. Tokenization returned `input_ids` and `attention_mask` for each statement-reason pair.

3. **Fine-tuning:**
   Training was performed using the `Trainer` API. Training arguments were specified with `TrainingArguments`, including the number of epochs, batch size, and learning rate. A `DataCollatorForMultipleChoice` was used to handle batches for the Multiple Choice format.

4. **Prediction:**
   After fine-tuning, predictions were obtained using the `Trainer.predict` method. For each statement, the predicted logits for all reasons were computed, and the reason with the highest logit was selected using `argmax`.

5. **Evaluation:**
   SubTask B was evaluated using accuracy. With a reduced dataset and base model, the expected accuracy was ~0.51. With full training, the accuracy reached approximately 0.928.

## Notes

* The notebook included settings for `shrink_dataset` and `base_model` to allow quick experimentation on local machines. For full training, these settings were set to `False` and training was recommended on a cloud platform such as [Google Colab](https://colab.research.google.com/) with GPU or TPU support.
* Random seeds were set to ensure reproducibility, though minor differences could occur due to hardware or software variations.
