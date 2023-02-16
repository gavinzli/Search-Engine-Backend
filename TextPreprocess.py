import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim import corpora
def TextPreprocess(content, add_bigrams=True, filterout_words=True):
  content["Content"] = content["Content"].fillna('').str.lower()
  docs = content["Content"].values.tolist()
  tokenizer = RegexpTokenizer(r'\w+')
  for idx in range(len(docs)):
    # Split into words.
    docs[idx] = tokenizer.tokenize(docs[idx])

  lemmatizer = WordNetLemmatizer()
  stop_words = stopwords.words('english')
  stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
  # Remove numbers, but not words that contain numbers.
  # Remove words that are only one character.
  # Lemmatize the documents.
  # Remove Stopwords
  docs = [[lemmatizer.lemmatize(token) for token in doc if not token.isnumeric() and len(token) > 1 and token not in stop_words] for doc in docs]

  # Compute bigrams.
  # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
  add_bigrams = True
  if add_bigrams == True:
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
      for token in bigram[docs[idx]]:
        if '_' in token:
          # Token is a bigram, add to document.
          docs[idx].append(token)
  id2word = corpora.Dictionary(docs)

  # Filter out words that occur less than 20 documents, or more than 50% of the documents.
  filterout_words = True
  if filterout_words == True:
    id2word.filter_extremes(no_below=20, no_above=0.5)

  # Bag-of-words representation of the documents.
  # The first element in the tuple is the id of the dictionary, the second element is the count of the document
  corpus = [id2word.doc2bow(doc) for doc in docs]
  return docs, id2word, corpus