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
1. [BBC Hindi News Dataset](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1)


## Results

#### Language Model Perplexity

| Architecture/Dataset | Hindi Wikipedia Articles - 172k | Hindi Wikipedia Articles - 55k |
|:--------:|:----:|:----:|
|   ULMFiT  |  34.06  |  35.87  |
|  TransformerXL |  26.09  |  34.78  |

**Note**: [Nirant](https://github.com/NirantK) has done previous [SOTA work with
 Hindi Language Model](https://github.com/NirantK/hindi2vec) and achieved perplexity of ~46.
  The scores above aren't directly comparable with his score because his train and test set
   were different and [test set isn't available for reproducibility](https://github.com/NirantK/hindi2vec/issues/1)
 

#### Classification Metrics

##### ULMFiT

| Dataset | Accuracy | Kappa Score |
|:--------:|:----:|:----:|
| Hindi Movie Reviews Dataset |  61.66  |  42.29  |
| BBC Hindi Dataset |  79.79  |  73.01  |

 
#### Visualizations
 
##### Embedding Space

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_30k.json) |
| TransformerXL | [Embeddings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/embedding_projector_config_transformerxl.json)  |

##### Sentence Encodings

| Architecture | Visualization |
|:--------:|:----:|
| ULMFiT | [Encodings projection](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/goru001/nlp-for-hindi/master/language-model/sentence_encodings/encoding_projector_config.json) |


## Pretrained Models

#### Language Models 
Download pretrained Language Models of ULMFiT, TransformerXL trained on
 [Hindi Wikipedia Articles - 172k and Hindi Wikipedia Articles - 55k](https://github.com/goru001/nlp-for-hindi#dataset)
  from [here](https://drive.google.com/open?id=1_8l5HFHHm4cboA-tkGbn3i6sfOWLmGyC)

#### Classifier

Download Movie Review classifier from [here](https://drive.google.com/open?id=1namfgTvH72Hgq3kPD8F43tOgLUEmG2zf)

Download BBC News classifier from [here](https://drive.google.com/open?id=1namfgTvH72Hgq3kPD8F43tOgLUEmG2zf)

#### Tokenizer

Unsupervised training using Google's [sentencepiece](https://github.com/google/sentencepiece)

Download the trained model and vocabulary from [here](https://drive.google.com/open?id=1TVuqY3Lad_KdY5Aj8ynGYVvoX5qgk2fJ)

