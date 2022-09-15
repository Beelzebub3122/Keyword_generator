import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category = Warning)
stopwords = stopwords.words('english')

def main():
    

    Excel = input('Path to input excel file:')
    out_excel = input('New name for output excel file:')
    df = pd.read_excel(Excel)


    # lemmatize complexity of words
    def lemmatization(texts, allowed_postags=['NOUN','VERB','ADJ', 'ADV']):
        nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
        text_out=[]
        for text in texts:
            doc=nlp(text)
            new_text=[]
            for token in doc:
                if token.pos_ in allowed_postags:
                    new_text.append(token.lemma_)
            final = ' '.join(new_text)
            text_out.append(final)
        return text_out

    lemmatized_text = lemmatization(df['tweet'])

    # generate elemental list
    def gen_words(texts):
        final=[]
        for text in texts:
            new = gensim.utils.simple_preprocess(text, deacc=True)
            final.append(new)
        return(final)

    data_words = gen_words(lemmatized_text)


    # create bi and tri grams
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold =100)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold =100)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    def make_bigrams(texts):
        return(bigram[doc] for doc in texts)

    def make_trigrams(texts):
        return (trigram[bigram[doc]] for doc in texts)

    data_bigrams = make_bigrams(data_words)
    data_bigrams_trigrams = make_trigrams(data_bigrams)

    x = list(data_bigrams_trigrams)

    X = []
    for i in x:
        for j in i:
            X.append(j) 
    Xdfx = pd.DataFrame(X)
    Xdfx.to_excel(out_excel)
    
if __name__=="__main__":
    main()