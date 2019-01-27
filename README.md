# nlp-for-hindi
State of the Art Tokenizer, Language model and Classifier for Hindi language (spoken in Indian sub-continent)


## Dataset
Download Wikipedia Articles Dataset (55,000 articles) which I scraped, cleaned and trained model on from [here](https://www.dropbox.com/sh/lz7t6fmkwq8ezke/AABuneKSfZoyCVF0DWTqMW4ja?dl=0).

`Note: There are more than 1.5 lakh Hindi wikipedia articles, whose urls you can find in the pickled object in the above folder. I chose to work with 55,000 articles only because of computational constraints.`

Get Hindi Movie Reviews Dataset which I scraped, cleaned and trained classification model from the repository path `datasets-preparation/hindi-movie-review-dataset`

Thanks to [nirantk](https://github.com/NirantK/hindi2vec/releases/tag/bbc-hindi-v0.1) for BBC Hindi News dataset 


## Results

Perplexity of Language Model: ~36 (on 20% validation set)

Accuracy of Movie Review classification model: ~53

Kappa Score of Movie Review classification model: ~30

Accuracy of BBC News classification model: ~79

Kappa Score of BBC News classification model: ~72

Note: [nirantk](https://github.com/NirantK/hindi2vec) has done previous SOTA work with Hindi Language Model and achieved perplexity of ~46. I have achieved better perplexity i.e ~35, but these scores aren't directly comparable because he used hindi wikipedia dumps for training whereas I scraped 55,000 articles and cleaned them through scripts in `datasets-preparation`. Though, one big reason I feel the results I have achieved should be better because
 I'm using `sentencepiece` for unsupervized tokenization whereas `nirantk`
 was using `spacy`.
 
 
## Pretrained Language Model

Download pretrained Language Model from [here](https://www.dropbox.com/sh/l9lm2rgsk7kupz6/AAAByHPkIvhHDgNDoq3v8yhoa?dl=0)

## Classifier

Download Movie Review classifier from [here](https://www.dropbox.com/sh/zdz6f4uu0rl0sxr/AADg-bA4D8JdYmg0QuECfB8Ra?dl=0)

Download BBC News classifier from [here](https://www.dropbox.com/sh/zdz6f4uu0rl0sxr/AADg-bA4D8JdYmg0QuECfB8Ra?dl=0)

## Tokenizer

Unsupervised training using Google's [sentencepiece](https://github.com/google/sentencepiece)

Download the trained model and vocabulary from [here](https://www.dropbox.com/sh/rbk9i53s33pgcqj/AAAvHTEBksEsWQsIrfbsh0-ba?dl=0)

