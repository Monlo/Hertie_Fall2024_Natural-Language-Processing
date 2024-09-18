# ---------------------------------------------------------------------------------------
# NATURAL LANGUAGE PROCESSING
# Session 3: Text Preprocessing
# Instructor: Sascha Göbel
# September 2024
# ---------------------------------------------------------------------------------------


# PREPARATIONS ==========================================================================

# load libraries ------------------------------------------------------------------------
from datasets import load_dataset
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
import sentencepiece as spm
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import os

# Change directory ---------------------------------------------------------------------------
os.chdir('02_Text preprocessing/')

# import datasets -----------------------------------------------------------------------
# https://huggingface.co/datasets/jonathanli/legal-advice-reddit
ds_reddit = load_dataset("jonathanli/legal-advice-reddit", split="train")
ds_reddit = ds_reddit.to_pandas().iloc[0:1000,:]
ds_reddit["text"] = ds_reddit["title"] + " " + ds_reddit["body"]

# https://huggingface.co/datasets/snats/url-classifications
ds_urls = load_dataset("snats/url-classifications", split="train")
ds_urls = ds_urls.to_pandas().iloc[0:1000,:]


# RULE-BASED TOKENIZATION ===============================================================

# splitting on whitespace ---------------------------------------------------------------
reddit_tokens_ws = ds_reddit["text"].str.split()
reddit_tokens_ws[0]
urls_tokens_ws = ds_urls["url"].str.split() # doesn't work
urls_tokens_ws[0]

# NLTK’s recommended word tokenizer -----------------------------------------------------
# improved TreebankWordTokenizer along with PunktSentenceTokenizer
reddit_tokens_nltk = ds_reddit["text"].map(word_tokenize)
reddit_tokens_nltk[6]

# sklearn tokenizer ---------------------------------------------------------------------
# It removes all the punctuation
reddit_tokens_sklearn = CountVectorizer(analyzer="word", lowercase=False)
reddit_tokens_sklearn.fit(ds_reddit["text"])
reddit_tokens_sklearn.get_feature_names_out()
reddit_tokens_sklearn = reddit_tokens_sklearn.build_analyzer()
reddit_tokens_sklearn(ds_reddit["text"][0])

# spaCy tokenizer -----------------------------------------------------------------------
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "ner", "lemmatizer", "attibute_ruler"])
reddit_tokens_spacy = ds_reddit["text"].map(nlp)
[tok.text for tok in reddit_tokens_spacy[0]]

# sklearn ngram tokenizer ---------------------------------------------------------------
reddit_tokens_sklearn_bg = CountVectorizer(analyzer="word", ngram_range=(2,2),
                                           lowercase=False)
reddit_tokens_sklearn_bg.fit(ds_reddit["text"])
reddit_tokens_sklearn_bg = reddit_tokens_sklearn_bg.build_analyzer()
reddit_tokens_sklearn_bg = ds_reddit["text"].map(reddit_tokens_sklearn_bg)
reddit_tokens_sklearn_bg[1]

# spaCy (or any other) tokenizer with sklearn -------------------------------------------
def custom_tokenizer(text): # wrap tokenizer in custom function
    tokenized_text = nlp(text)
    return [tok.text for tok in tokenized_text]
reddit_tokens_custom = CountVectorizer(analyzer="word", ngram_range=(2,2),
                                       tokenizer=custom_tokenizer,
                                       lowercase=False)
reddit_tokens_custom.fit(ds_reddit["text"])
reddit_tokens_custom = reddit_tokens_custom.build_analyzer()
reddit_tokens_custom = ds_reddit["text"].map(reddit_tokens_custom)
reddit_tokens_custom[1]


# DATA-DRIVEN TOKENIZATION ==============================================================

# Byte-Pair encoding --------------------------------------------------------------------
tokenizer_bpe = Tokenizer(BPE(unk_token="[UNK]")) # Initialize the tokenizer
tokenizer_bpe.pre_tokenizer = pre_tokenizers.Whitespace()
trainer_bpe = BpeTrainer()
tokenizer_bpe.train_from_iterator(ds_reddit["text"], trainer_bpe)
tokenizer_bpe.encode("Here is some new text, can you parse this #boringstuff").tokens

# WordPiece encoding --------------------------------------------------------------------
tokenizer_wp = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer_wp.pre_tokenizer = pre_tokenizers.Whitespace()
trainer_wp = WordPieceTrainer()
tokenizer_wp.train_from_iterator(ds_reddit["text"], trainer_wp)
tokenizer_wp.encode("Here is some new text, can you parse this #boringstuff").tokens

# SentencePiece encoding ----------------------------------------------------------------
ds_reddit["text"].to_csv("reddit.txt", sep="\n", index=False)
spm.SentencePieceTrainer.train(input="reddit.txt",
                               model_prefix='sentencePiece_model',
                               model_type="unigram",
                               vocab_size=10929)
tokenizer_sp = spm.SentencePieceProcessor(model_file="./sentencePiece_model.model")
tokenizer_sp.encode_as_pieces("Here is some new text, can you parse this #boringstuff")

ds_urls["url"].to_csv("urls.txt", sep="\n", index=False)
spm.SentencePieceTrainer.train(input="urls.txt",
                               model_prefix='sentencePiece_model',
                               model_type="unigram",
                               vocab_size=5886)
tokenizer_sp = spm.SentencePieceProcessor(model_file="./sentencePiece_model.model")
tokenizer_sp.encode_as_pieces(ds_urls["url"][1])
tokenizer_sp.encode_as_pieces("https://www.huggingface.co/datasets/snats/url-classifications")
tokenizer_sp.encode_as_pieces("https://www.google.de")
tokenizer_sp.encode_as_pieces("https://www.wikipedia.org")


# NORMALIZATION =========================================================================

# case folding --------------------------------------------------------------------------
reddit_cf = ds_reddit["text"].str.lower()

# stopword removal ----------------------------------------------------------------------
nltk.download("stopwords")
stop_words_en = nltk.corpus.stopwords.words("english")
len(stop_words_en)
print(stop_words_en)

def remove_stopwords(text, stop_words):
    new_text = [token for token in text if token not in stop_words]
    return new_text

reddit_tokens_cf = reddit_cf.map(word_tokenize)
reddit_tokens_cf_sw = reddit_tokens_cf.apply(remove_stopwords, stop_words=stop_words_en)
len(reddit_tokens_cf_sw[0])
len(reddit_tokens_cf[0])

# stemming ------------------------------------------------------------------------------
stemmer = nltk.stem.porter.PorterStemmer()

def stem_words(text, stemmer):
    new_text = [stemmer.stem(token) for token in text]
    return new_text

reddit_tokens_cf_sw_st = reddit_tokens_cf_sw.apply(stem_words, stemmer=stemmer)
reddit_tokens_cf_sw_st[0]

# lemmatization -------------------------------------------------------------------------
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_words(text, lemmatizer, pos):
    new_text = [lemmatizer.lemmatize(token, pos=pos) for token in text]
    return new_text

reddit_tokens_cf_sw_lt = reddit_tokens_cf_sw.apply(lemmatize_words, lemmatizer=lemmatizer,
                                                   pos="v")
reddit_tokens_cf_sw_lt[0]
reddit_tokens_cf_sw[0]
