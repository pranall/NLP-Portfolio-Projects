# Sequence Labeling for Quantity Recognition in Scientific Texts

This project addresses the **Quantity** recognition step of the MeasEval shared task from SemEval-2021. The goal is to identify spans containing quantities (e.g., `12 kg`) in scientific documents. The task is approached as a **Sequence Labeling** problem using a Recurrent Neural Network (RNN) implemented with **Keras** and **TensorFlow**.

---

The following reference links were used to create this project:

• [Layer weight initializers](https://keras.io/api/layers/initializers/)
• [Using pre-trained word embeddings Doc](https://keras.io/examples/nlp/pretrained_word_embeddings/)
• [NumPy Documentation](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
• [Model training APIs Documentation](https://keras.io/api/models/model_training_apis/#predict-method)
• [TimeDistributed layer Document](https://keras.io/api/layers/recurrent_layers/time_distributed/)
• [Dense Layer Documentation](https://keras.io/api/layers/core_layers/dense/)
• [Bidirectional Layer Documentation](https://keras.io/api/layers/recurrent_layers/bidirectional/)
• [LSTM Layer Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
• [Embedding Layer Documentation](https://keras.io/api/layers/core_layers/embedding/)
• [The Sequential Model](https://keras.io/guides/sequential_model/)
• [Making new layers and models via subclassing Documentation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
• [The functional API Documentation](https://keras.io/guides/functional_api/)
• [Pad Sequences Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences)
• [API Reference](https://pandas.pydata.org/docs/reference/index.html)
• [Tensor Flow documentation](https://www.tensorflow.org/)
• [Keras Documentation](https://keras.io/)

----

## Dataset

[GloVe pre-trained word vectors Dataset](https://nlp.stanford.edu/projects/glove/)

The dataset consists of token-level annotations following a BIO schema with three labels: `B-Quantity`, `I-Quantity`, and `O`.

* **Train set:** 248 articles, 1366 sentences
* **Development set:** 65 documents, 459 sentences
* **Test set:** 136 articles, 848 sentences

Each token has a corresponding lemma, which is used as input for the model.

---

## Data Pre-processing

1. **Vocabulary and Label Mapping:**

   * Unique lemmas were collected and special tokens `[PAD]` and `[UNK]` were added.
   * Label set includes `[PAD]`, `O`, `B-Quantity`, `I-Quantity`.
   * Dictionaries mapping lemmas and labels to indices were created.
   * Vocabulary size: 5508
   * Labels size: 4

2. **Sentence Integration:**

   * Tokens and labels were aggregated at the sentence level.

3. **Numerical Representation:**

   * Sentences were converted to sequences of lemma indices.
   * Labels were converted to sequences of label indices.
   * All sequences were padded to a fixed length (`maxlen = 130`) to form uniform input tensors.

---

## Model Architecture

A **Sequential RNN** was implemented with the following layers:

1. **Embedding Layer:**

   * Input dimension equal to vocabulary size
   * Output dimension: `embedding_dim`
   * Padding masked to ignore `[PAD]` tokens

2. **Bidirectional LSTM:**

   * Units: `rnn_units`
   * Returns full sequences for token-level predictions

3. **TimeDistributed Dense Layer:**

   * Units equal to number of labels
   * `softmax` activation for multi-class prediction

The model was compiled with:

* Loss: `sparse_categorical_crossentropy`
* Optimizer: `adam`
* Metrics: `sparse_categorical_accuracy`

---

## Training

* Hyperparameters were set for reproducibility by initializing the random seed.
* A `shrink_dataset` option allowed faster debugging with a subset of the training data.
* The model was trained using the full training set for final evaluation.

---

## Predictions and Evaluation

* Predictions were obtained as sequences of label indices and converted back to label strings.
* The evaluation metric approximates **Reading Comprehension Macro-Averaged F1**, measuring the overlap of predicted and true quantity tokens.

**Results (Randomly Initialized Embeddings):**

| Label     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Quantity  | 0.85      | 0.69   | 0.76     | 1263    |
| micro avg | 0.85      | 0.69   | 0.76     | 1263    |
| macro avg | 0.85      | 0.69   | 0.76     | 1263    |

---

## Pre-trained Word Embeddings

* **GloVe embeddings** were used to initialize the Embedding layer.
* The embedding matrix was constructed with dimensions `(vocab_size, embedding_dim)` and matched each lemma to its GloVe vector if available.
* The Embedding layer was trainable, allowing fine-tuning during training.

**Results with GloVe Embeddings:**

| Label     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Quantity  | 0.85      | 0.79   | 0.82     | 1263    |
| micro avg | 0.85      | 0.79   | 0.82     | 1263    |
| macro avg | 0.85      | 0.79   | 0.82     | 1263    |

**Impact:** Incorporating pre-trained embeddings improved recall and F1-score, enabling better identification of quantities while maintaining precision.

---

## Tools and Libraries

* ipython==8.5.0
* jupyter==1.0.0
* nbimporter==0.3.4
* pytest==7.1.3
* pandas==1.3.5
* spacy==3.2.4
* tensorflow==2.9.2
* numpy==1.24.2
* scikit-learn==1.2.0
---------
## Requirements for MACOS

* ipython==8.5.0
* jupyter==1.0.0
* nbimporter==0.3.4
* pytest==7.1.3
* pandas==1.3.5
* spacy==3.2.4
* tensorflow-macos==2.9.2
* numpy==1.24.2
* scikit-learn==1.2.0

