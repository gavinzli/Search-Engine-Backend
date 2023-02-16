from flask import Flask, request
import json
from flask_jsonpify import jsonpify
import psycopg2
from flask_restful import Resource, Api
from apispec import APISpec
from marshmallow import Schema, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_apispec.extension import FlaskApiSpec
from flask_apispec.views import MethodResource
from flask_apispec import marshal_with, doc, use_kwargs
from TextPreprocess import TextPreprocess
# from googletrans import Translator
import os
import pandas as pd
import numpy as np
# import sqlalchemy as db
from gensim import models
from gensim.matutils import cossim
import pickle
from threading import Thread
import warnings
warnings.filterwarnings("ignore")
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem.wordnet import WordNetLemmatizer
NUMBER_OF_TOPICS = 10

# custom thread
class TopicalThread(Thread):
    # constructor
    def __init__(self,topical_input):
        # execute the base constructor
        Thread.__init__(self)
        self.topical_input = topical_input
        # set a default value
        self.value = None
    # function executed in a new thread
    def run(self):
        topical_sim = [cossim(self.topical_input, doci) for doci in topical_docs]
        self.value = topical_sim

class TextualThread(Thread):
    # constructor
    def __init__(self,text_input):
        # execute the base constructor
        Thread.__init__(self)
        self.text_input = text_input
        # set a default value
        self.value = None
    # function executed in a new thread
    def run(self):
        input_tfidf = vectorizer.transform([self.text_input])
        input_tfidf.resize((1, len(vectorizer.get_feature_names_out())))
        input_df = pd.DataFrame(input_tfidf.T.todense(), index=vectorizer.get_feature_names_out()).reset_index()
        input_df.drop(columns=['index'], inplace=True)
        input_df = input_df.loc[input_df[0] != 0]
        input_vec = list(input_df.to_records())
        textual_sim = []
        for ind, column in enumerate(origin_df.columns):
            textual_sim.append(cossim(origin_df[[column]].loc[origin_df[column] != 0].to_records(), input_vec))
        self.value = textual_sim

class CrowdRankThread(Thread):
    # constructor
    def __init__(self,papers):
        # execute the base constructor
        Thread.__init__(self)
        self.papers = papers
        # set a default value
        self.value = None
    # function executed in a new thread
    def run(self):
        p = self.papers
        ref_df = reference[['SourceContentID', 'ReferenceContentID']].merge(
            right=p[['id', 'topical_sim']],
            left_on=['SourceContentID'],
            right_on=['id'],
            how='left'
        )
        # generate LinkCount and sum of
        crowd_rank_df = ref_df.groupby('ReferenceContentID').agg({'SourceContentID': 'count', 'topical_sim': 'sum'})\
            .reset_index().rename(columns={'ReferenceContentID': 'id', 'SourceContentID': 'LinkCount'})
        crowd_rank_df['CrowdRank'] = crowd_rank_df['topical_sim'] / (np.log(crowd_rank_df['LinkCount'] + 1) + 1)

        # merge CrowdRank value
        p = p.merge(crowd_rank_df[['id', 'CrowdRank']], left_on=['id'], right_on=['id'], how='left')
        p['CrowdRank'] = p['CrowdRank'].fillna(0)
        # # min-max normalization
        # scaler = MinMaxScaler()
        # normalized_data = scaler.fit_transform(papers['CrowdRank'])
        p['CrowdRank'] = (p['CrowdRank'] - p['CrowdRank'].min()) / (p['CrowdRank'].max() - p['CrowdRank'].min())
        self.value = p


def fetch_table(table_name):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM {};".format(table_name))
    table = cur.fetchall()
    cur.close()
    conn.close()
    return table

def get_db_connection():
    conn = psycopg2.connect(host='34.168.68.153',
                            database='postgres',
                            user='vm',
                            password='F*JL(oL<$p%.bM{i')
    return conn
# Load Source Data
# content = fetch_table("content")
# reference = fetch_table("reference")
content = pd.DataFrame(fetch_table("content"), columns=['id', 'Source', 'Category', 'Title', 'Keyword', 'Subtitle', 'Authors',
       'Content', 'PublishDate', 'Hyperlink', 'NoOfView'])
reference = pd.DataFrame(fetch_table("reference"),columns=['RefID', 'hyperlink', 'SourceContentID', 'ReferenceContentID'])
# Load LDA Model
lda = models.ldamodel.LdaModel.load('Model/lda_model')
texts, id2word, corpus = TextPreprocess(pd.DataFrame(content))
topical_docs = [lda.get_document_topics(i) for i in corpus]
combine_docs = []
for idx in range(len(texts)):
    combine_docs.append(' '.join(map(str, texts[idx])))

vectorizer = pickle.load(open("Model/vectorizer.pickle", "rb"))
origin_tfidf = vectorizer.transform(combine_docs)
origin_df = pd.DataFrame(origin_tfidf.T.todense(), index=vectorizer.get_feature_names_out()).reset_index()
origin_df.drop(columns=['index'], inplace=True)

LogRegModel = pickle.load(open("Model/LogRegModel.sav", 'rb'))

id2wordfile = pickle.load(open('Model/id2word.sav', 'rb'))

app = Flask(__name__)  # Flask app instance initiated
api = Api(app)  # Flask restful wraps Flask app around it.
app.config.update({
    'APISPEC_SPEC': APISpec(
        title='Search Engine',
        version='v1',
        plugins=[MarshmallowPlugin()],
        openapi_version='2.0.0'
    ),
    'APISPEC_SWAGGER_URL': '/swagger/',  # URI to access API Doc JSON
    'APISPEC_SWAGGER_UI_URL': '/swagger-ui/'  # URI to access UI of API Doc
})
docs = FlaskApiSpec(app)


class SearchResponseSchema(Schema):
    message = fields.Str(dump_default='Success',
                         metadata={'description': 'Return Example'})

class SearchRequestSchema(Schema):
    text = fields.String(
        required=True,
        metadata={'description': 'API type of awesome API'})
    
# class TranslationResponseSchema(Schema):
#     message = fields.Str(dump_default='Success',
#                          metadata={'description': 'Translation Result'})

# class TranslationRequestSchema(Schema):
#     text = fields.String(
#         required=True,
#         metadata={'description': 'Translation Text'})


#  Restful way of creating APIs through Flask Restful
class Search(MethodResource, Resource):
    @doc(description='Search Result.', tags=['Search'])
    @use_kwargs(SearchRequestSchema, location=('json'))
    @marshal_with(SearchResponseSchema)  # marshalling
    def post(self, **kwargs):
        # starttime = time.perf_counter()
        p = content
        input_docs = request.get_json()['text']
        combine_input = id2wordfile.doc2bow(input_docs.lower().split())
        # text_input, combine_input = fitPreprocess(input_docs, id2wordfile)
        topical_input = lda.get_document_topics(combine_input)
        topical_thread = TopicalThread(topical_input)
        textual_thread = TextualThread(input_docs)
        textual_thread.start()
        topical_thread.start()
        textual_thread.join()
        p['topical_sim'] = topical_thread.value
        crowdRank_thread = CrowdRankThread(p)
        crowdRank_thread.start()
        crowdRank_thread.join()
        p = crowdRank_thread.value
        topical_thread.join()
        p['textual_sim'] = textual_thread.value
        p['sum'] = p['topical_sim'] * LogRegModel.coef_[0, 0] + p['textual_sim'] * LogRegModel.coef_[0, 1] + p['CrowdRank'] * LogRegModel.coef_[0, 1]
        # print(time.perf_counter()-starttime)
        p = p.sort_values('sum', ascending=False)
        return jsonpify(p.to_dict(orient="records"))
            
        '''
        Get method represents a GET API method
        '''
        # return {'message': 'My First API'}
        # print(pd.DataFrame(content).to_json())
        # return pd.DataFrame(content).to_json(orient="records")
    
# class Translation(MethodResource, Resource):
#     @doc(description='Translation.', tags=['Translation'])
#     @use_kwargs(TranslationRequestSchema, location=('json'))
#     @marshal_with(TranslationResponseSchema)  # marshalling
#     def post(self, **kwargs):
#         translator = Translator()
#         return {'message': translator.translate(request.get_json()['text'], src='zh-cn', dest='en').text}

api.add_resource(Search, '/Search')
docs.register(Search)
# api.add_resource(Translation, '/Translation')
# docs.register(Translation)

if __name__ == '__main__':
    app.run(debug=True)
