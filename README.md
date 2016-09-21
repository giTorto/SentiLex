# SentiLex
Simple Lexicon-based Sentiment Analysis

## Dependencies
The script in order to provide lemmatization and tokenization is based on  wordnet corpus from nltk.
In addition it expects that lexicon files are provided.

## Usage example
```python
from SentiLex import SentiSentenceAnalyser
shifter_file_name = '../resources/en-shifters.tsv'
intensier_file_name= '../resources/en-intensifiers.tsv'
polarity_lex_file_name = '../resources/en-lexicon.tsv'
doc_str = "Don't be depressed"
sentence_sentiment_analyser = SentiSentenceAnalyser(shifters_file=shifter_file_name,
                                                    intens_file=intensifier_file_name,
													lexicon_file=polarity_lex_file_name,
													token_sep='+',
													discountinuous_entry_sep='*',
													field_separator='\t',
													newer_version=False, tweet_tokenizer=True)
score,sent = sentence_sentiment_analyser.sentence_to_sentiment_score(doc_str,lemmatization=False)
```