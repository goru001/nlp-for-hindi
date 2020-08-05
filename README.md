# NLP for Hindi
This repository contains State of the Art Language models and Classifier for Hindi language
 (spoken in Indian sub-continent).
  
The models trained here have been used in [Natural Language Toolkit for Indic Languages
 (iNLTK)](https://github.com/goru001/inltk)


## Dataset

#### Created as part of this project
1. [Hindi Wikipedia Articles - 172k](https://www.kaggle.com/disisbig/hindi-wikipedia-articles-172k)

2. [Hindi Wikipedia Articles - 55k](https://www.kaggle.com/disisbig/hindi-wikipedia-articles-55k)

3. [Hindi Movie Reviews Dataset](https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset)

4. [Hindi Text Short Summarization Corpus](https://www.kaggle.com/disisbig/hindi-text-short-summarization-corpus)

5. [Hindi Text Short and Large Summarization Corpus](https://www.kaggle.com/disisbig/hindi-text-short-and-large-summarization-corpus)


#### Open Source Datasets
1. [BBC News Articles](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for Hindi documents extracted from BBC news website.

2. [IIT Patna Product Reviews](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for product reviews posted in Hindi.

3. [IIT Patna Movie Reviews](https://github.com/ai4bharat-indicnlp/indicnlp_corpus#publicly-available-classification-datasets) : Sentiment analysis corpus for movie reviews posted in Hindi.

## Results

### Language Model Perplexity (on validation set)

| Architecture/Dataset | Hindi Wikipedia Articles - 172k | Hindi Wikipedia Articles - 55k |
|:--------:|:----:|:----:|
|   ULMFiT  |  34.06  |  35.87  |
|  TransformerXL |  26.09  |  34.78  |

**Note**: [Nirant](https://github.com/NirantK) has done previous [SOTA work with
 Hindi Language Model](https://github.com/NirantK/hindi2vec) and achieved perplexity of ~46.
  The scores above aren't directly comparable with his score because his train and validation set
   were different and [they aren't available for reproducibility](https://github.com/NirantK/hindi2vec/issues/1)
 

### Classification Metrics

##### ULMFiT

| Dataset | Accuracy | MCC | Notebook to Reproduce results |
|:--------:|:----:|:----:|:----:|
| BBC News Articles |  78.75  |  71.61  | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_BBC_Articles.ipynb) |
| IIT Patna Movie Reviews | 57.74 | 37.23 | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_IITP%2BMovie.ipynb) |
| IIT Patna Product Reviews |  75.71  |  59.76  | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_IITP_Product.ipynb) |
 
 

### Visualizations
 
##### Word Embeddings

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_30k.json) |
| TransformerXL | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_transformerxl.json)  |

##### Sentence Embeddings

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Encodings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/sentence_encodings/encoding_projector_config.json) |




### Results of using Transfer Learning + Data Augmentation from iNLTK

##### On using complete training set (with Transfer learning)

| Dataset | Dataset size (train, valid, test) | Accuracy | MCC | Notebook to Reproduce results |
|:--------:|:----:|:----:|:----:|:----:|
| IIT Patna Movie Reviews | (2480, 310, 310) | 57.74 | 37.23 | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_IITP%2BMovie.ipynb) |
 

##### On using 20% of training set (with Transfer learning)

| Dataset | Dataset size (train, valid, test) | Accuracy | MCC | Notebook to Reproduce results |
|:--------:|:----:|:----:|:----:|:----:|
| IIT Patna Movie Reviews | (496, 310, 310) | 47.74 | 20.50 | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_IITP%2BMovie_without_Data_Aug.ipynb) |
 
##### On using 20% of training set (with Transfer learning + Data Augmentation)

| Dataset | Dataset size (train, valid, test) | Accuracy | MCC | Notebook to Reproduce results |
|:--------:|:----:|:----:|:----:|:----:|
| IIT Patna Movie Reviews | (496, 310, 310) | 56.13 | 34.39 | [Link](https://github.com/goru001/nlp-for-hindi/blob/master/classification-benchmarks/Hindi_Classification_Model_IITP%2BMovie_with_Data_Aug.ipynb) |


## Pretrained Models

#### Language Models 
Download pretrained Language Models of ULMFiT, TransformerXL trained on
 [Hindi Wikipedia Articles - 172k and Hindi Wikipedia Articles - 55k](https://github.com/goru001/nlp-for-hindi#dataset)
  from [here](https://drive.google.com/open?id=1_8l5HFHHm4cboA-tkGbn3i6sfOWLmGyC)

#### Tokenizer

Unsupervised training using Google's [sentencepiece](https://github.com/google/sentencepiece)

Download the trained model and vocabulary from [here](https://drive.google.com/open?id=1TVuqY3Lad_KdY5Aj8ynGYVvoX5qgk2fJ)

