# ComVE SubTask A – Pre-trained Language Model Fine-Tuning

This project focuses on **SubTask A** of the [ComVE](https://competitions.codalab.org/competitions/21080) shared task from SemEval-2020, which evaluates whether a system can identify which of two similar natural language statements is nonsensical.

## Project Description

Given two statements that differ by only a few words, the goal is to select the one that does not make sense. For example:

* **Statement 1:** He put a turkey into the fridge.
* **Statement 2:** He put an elephant into the fridge.

Here, Statement 2 is nonsensical. This can be approached as a **Text Matching** problem, where the model predicts a label indicating which statement is nonsensical.

## Dataset

The **ComVE** dataset contains:

* **Train:** 10,000 statement pairs
* **Development:** 997 pairs
* **Test:** 1,000 pairs

Each pair is labeled `0` if the first statement is nonsensical, or `1` if the second is nonsensical.

The dataset can be loaded into `Datasets` objects and pre-processed using the Hugging Face [Datasets](https://huggingface.co/docs/datasets/index) library, including tokenization of the statement pairs with the model’s tokenizer.

## Model

This project uses **RoBERTa**, a Pre-trained Language Model based on BERT architecture, fine-tuned for sequence classification with the [Transformers](https://huggingface.co/docs/transformers/index) library.

### Steps:

1. **Load Pre-trained Model:** The model and tokenizer are loaded using `AutoModelForSequenceClassification` and the corresponding tokenizer.
2. **Pre-processing:** Statement pairs are tokenized, padded, and truncated to a fixed `max_length`. The tokenizer produces `input_ids` and `attention_mask` arrays.
3. **Trainer Creation:** The Hugging Face `Trainer` API is used to fine-tune the model on the train dataset and evaluate on the development dataset.
4. **Prediction:** After training, the `Trainer.predict` method produces logits for each example. Predictions are converted to labels by selecting the class with the highest logit.
5. **Evaluation:** Accuracy is computed by comparing predictions with true labels using the Hugging Face [evaluate](https://huggingface.co/docs/evaluate/index) library.

### Example Prediction

For the statement pair:

* **Statement 1:** a duck walks on three legs
* **Statement 2:** a duck walks on two legs

The model outputs logits `[0.1005, -0.0192]`, so the predicted label is `0`, indicating Statement 1 is nonsensical.

## Hyperparameters and Training

* Fine-tuning uses configurable hyperparameters including `epochs`, `batch_size`, and `learning_rate`.
* For quick experimentation, a reduced dataset and base model can be used (`shrink_dataset=True`, `base_model=True`). Full training (`shrink_dataset=False`, `base_model=False`) achieves significantly higher performance.

## Results

* **Reduced Training:** Accuracy ≈ 0.49
* **Full Training:** Accuracy ≈ 0.929

## Notes

* Reproducibility is controlled via a fixed random seed, though exact results may vary depending on hardware and software versions.
* Large-scale fine-tuning may require GPU or TPU support, e.g., using [Google Colab](https://colab.research.google.com/).

This setup allows efficient fine-tuning of Pre-trained Language Models for commonsense reasoning tasks like ComVE SubTask A.
