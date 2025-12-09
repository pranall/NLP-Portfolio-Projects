# Pre-trained Language Models: SubTask C

This repository contains the implementation of **SubTask C** from the [ComVE](https://competitions.codalab.org/competitions/21080) shared task, part of SemEval-2020. The task evaluates whether a system can determine if a natural language statement makes sense and generate a reason explaining why it does not.

## Task Description

SubTask C required generating a valid reason for a nonsensical statement. Each nonsensical statement came with three reference reasons, for example:

**Statement:** He put an elephant into the fridge.
**Reason A:** An elephant is much bigger than a fridge.
**Reason B:** A fridge is much smaller than an elephant.
**Reason C:** Most of the fridges aren't large enough to contain an elephant.

This subtask was approached as a sequence-to-sequence problem, where the input was the nonsensical statement and the output was a valid reason.

## Dataset

* **Training set:** 10,000 nonsensical statements with three reference reasons each.
* **Development set:** 997 nonsensical statements with three reference reasons each.
* **Test set:** 1,000 nonsensical statements with three reference reasons each.

The train and development datasets contained the following columns:

* `id`: statement identifier
* `FalseSent`: the nonsensical statement
* `reason`: one of the reference reasons

The test dataset contained:

* `id`: statement identifier
* `FalseSent`: the nonsensical statement
* `reason1`, `reason2`, `reason3`: reference reasons

## Pre-trained Model

The notebook fine-tuned a [BART](https://huggingface.co/docs/transformers/model_doc/bart) pre-trained sequence-to-sequence model using the [Transformers](https://huggingface.co/docs/transformers/index) library. BART was selected for its ability to generate coherent sequences and its architecture optimized for sequence-to-sequence tasks.

### Dataset Shrinking

For quick experimentation, a reduced dataset (`shrink_dataset=True`) and the base version of BART (`base_model=True`) were used. For full-scale training, the full dataset and model were used with `shrink_dataset=False` and `base_model=False`.

### Hardware Considerations

Fine-tuning a pre-trained language model requires significant computational resources. The notebook was executed in [Google Colab](https://colab.research.google.com/) with GPU acceleration.

## Pre-processing

* The tokenizer was applied to the `FalseSent` column for the test dataset.
* For the train and development datasets, the tokenizer was also applied to the `reason` column, and the resulting `input_ids` were stored in the `labels` field.
* All sequences were padded and truncated to a `max_length` hyperparameter.

## Training

* Training was conducted using `Seq2SeqTrainer` from the Transformers library.
* `Seq2SeqTrainingArguments` were configured to evaluate the model on the development set after each epoch.
* Checkpoints were disabled by setting `save_strategy="no"` to reduce storage requirements.

## Prediction

* The `Trainer` generated sequences of token indices for the test dataset.
* Token indices were decoded using the tokenizer to obtain the predicted text strings.
* Predictions were stored in the `prediction` column of the test DataFrame.

## Evaluation

* **Metrics:** BLEU and ROUGE scores were used to evaluate the generated reasons.
* **Expected scores with `shrink_dataset=True` and `base_model=True`:** BLEU ≈ 0.216, ROUGE ≈ 0.446
* **Expected scores with full training:** BLEU ≈ 0.228, ROUGE ≈ 0.461
