# Applied NLP Text Processing Pipeline

This project implemented a complete workflow for acquiring, cleaning, and annotating text from an online article using Python, BeautifulSoup, spaCy, and pandas. The pipeline extracted content from a downloaded HTML file, removed noise, applied linguistic pre-processing, and organized the resulting annotations into a structured DataFrame. A custom tokenizer was also integrated to address date-splitting requirements not handled by spaCy’s default configuration.

The following links were useful in building this project:

• [Import Requests URL](https://www.fao.org/newsroom/detail/world-food-prices-dip-in-december/en)

• [Python Built-in types](https://docs.python.org/3.9/library/stdtypes.html?highlight=replace#str)

• [Spacy](https://spacy.io/)

• [Language Processing Pipelines](https://spacy.io/usage/processing-pipelines)

• [Linguistic Features](https://spacy.io/usage/linguistic-features)

• [English Pipeline optimized for CPU](https://spacy.io/models/en#en_core_web_sm)

• [Beautiful Soup Documentation](https://beautiful-soup-4.readthedocs.io/en/latest/index.html)

## 1. Text Extraction

The HTML document included extensive tags, scripts, and navigation elements unrelated to the article. The `extract_text` function parsed the document with **BeautifulSoup**, located the element containing the article body, and extracted its text. The resulting output began with the article title and aligned with the expected reference excerpt.

## 2. Text Cleanup

The extracted text contained irregular spacing and multiple newline characters. The `clean_text` function normalized the text by removing redundant whitespace and appending missing sentence-final periods. The cleaned output produced a continuous, structured version of the article consistent with the defined format.

## 3. Linguistic Pre-processing

The cleaned text was processed with the **spaCy** `en_core_web_sm` pipeline. The pipeline provided:

* Sentence segmentation
* Tokenization
* Lemmatization
* Stopword identification
* Part-of-speech tagging
* Dependency labels
* Named entity recognition

The output served as the basis for downstream annotation.

## 4. DataFrame Construction

A **pandas** DataFrame was generated from the spaCy `Doc` object. Each token was represented as one row and included:

* `sent_id` – sentence index
* `token_id` – token index within the sentence
* `text` – token text
* `lemma` – lemmatized form
* `pos` – part-of-speech tag
* `ent` – named entity label

The structure reproduced the expected annotations for all sentences, including samples such as the FAO Food Price Index line.

## 5. Custom Tokenizer

The default spaCy tokenizer did not split dates formatted as `month/day/year`. The `customize_tokenizer` function updated the pipeline by extending the tokenizer’s **infix rules** to include a pattern for the slash (`/`). All other tokenizer components vocabulary, prefixes, infixes, suffixes, and matching rules were preserved.
With the updated infix pattern, dates such as `06/01/2023` were correctly segmented into the components `06`, `/`, `01`, `/`, `2023`.

---

This project demonstrates a complete, modular NLP workflow suitable for applied language-processing tasks involving text extraction, normalization, linguistic annotation, and structured data representation.

