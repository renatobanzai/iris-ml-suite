# -*- coding: utf-8 -*-
import dash
import dash_table
from pandas import DataFrame
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re, string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import urllib.request
from bs4 import BeautifulSoup
import jaydebeapi
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

jdbc_server = "jdbc:IRIS://iris-ml:51773/PYTHON"
jdbc_driver = 'com.intersystems.jdbc.IRISDriver'
iris_jdbc_jar = "./intersystems-jdbc-3.1.0.jar"
iris_user = "_SYSTEM"
iris_password = "SYS"

conn = jaydebeapi.connect(jdbc_driver, jdbc_server, [iris_user, iris_password], iris_jdbc_jar)


# load the model from disk
filename = 'predictors.sav'
predictors = pickle.load(open(filename, 'rb'))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

filename = 'vec.sav'
vec = pickle.load(open(filename, 'rb'))

filename = 'vec_integratedml.sav'
vec_integratedml = pickle.load(open(filename, 'rb'))

filename = 'vec_integratedml_bin.sav'
vec_integratedml_bin = pickle.load(open(filename, 'rb'))



# load tags
with open('all_tags.json') as json_file:
    all_tags = set(json.load(json_file))

# load formated
with open('formated_columns.json') as json_file:
    formated_columns = set(json.load(json_file))

# load formated
with open('predict_tags.json') as json_file:
    predict_tags = set(json.load(json_file))


def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def load_fromurl(pURL):
    page = urllib.request.urlopen(pURL)
    soup = BeautifulSoup(page)
    title = soup.head.title.string
    text = soup.find_all("div", class_="field-item")[0].get_text()
    return title + " " + text


view_query = []
for tag in all_tags:
    view_query.append(" SELECT top 20 id, Name, Tags, Text, PostType "
                      "FROM Community.Post "
                      "Where lang = 'en' "
                      "and tags like '%" + tag + "%' ")


str_view_query = " union ".join(view_query)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

def get_post_tag_classifier_1():
    return html.Div(children=[
        html.H1(children='Post Tag Classifier Using IRIS + ScikitLearn'),
        html.Div(children=[
            dcc.Input(id="txt_url",style={'width': '60%', "margin-right":"20px"},
                      placeholder="Development Community Post URL"),
            html.Button('Load Post from URL', id='load-url-button', n_clicks=0),
            dcc.Textarea(id='post-classifier-1',
                         style={'width': '90%', 'height': 200, "margin-top":"20px"},
                         placeholder="Text do Predict Tags (you can type or load by URL)"),
            html.Button('Predict Python SkLearn', id='predict-button-classifier-1',
                        style={"margin-right":"20px"}, n_clicks=0),
            html.Button('Predict IRIS IntegratedML', id='predict-button-classifier-2', n_clicks=0),
            html.Div(children=[
                html.Label('SkLearn Prediction:'),
                dcc.Dropdown(id='result-classifier-1',
                             multi=True,
                             style={'width': '90%'},
                             options=[{"label": opt, "value":opt} for opt in all_tags],
                             disabled=True
                             ),
                html.Label('IRIS IntegratedML Prediction:'),
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=dcc.Dropdown(id='result-classifier-2',
                                 multi=True,
                                 style={'width': '90%'},
                                 options=[{"label": opt, "value": opt} for opt in all_tags],
                                          disabled=True
                                 )
                )

        ])])
        ])




@app.callback(
    Output('result-classifier-1', 'value'),
    [Input('predict-button-classifier-1', 'n_clicks')],
    [State('post-classifier-1', 'value')])
def predict_classifier_1(n_clicks, post):
    #vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
    #                      min_df=1, max_df=1, strip_accents='unicode', use_idf=1,
    #                      smooth_idf=1, sublinear_tf=1)
    result = []
    if n_clicks > 0:
        df_post = DataFrame(columns=["text"])
        df_post["text"] = [clean_text(post.lower())]
        post_prepared = vec.transform(df_post["text"])
        for tag in all_tags:
            if predictors[tag].predict(post_prepared)[0] == 1:
                result.append(tag)

    return result


def get_integratedml_prediction(ptag, str_columns, curs):
    try:
        model_name = "has_{}_tag".format(ptag)
        sel = "SELECT PREDICT({} WITH {}) as result".format(model_name, str_columns)
        curs.execute(sel)
        iris_tags = curs.fetchall()
        result = 0
        if len(iris_tags) > 0:
            if iris_tags[0][0]=="1":
                result = 1

        return result
    except:
        return 0

@app.callback(Output("loading-output-1", "children"), [Input("input-1", "value")])
def input_triggers_spinner(value):
    time.sleep(1)
    return value


@app.callback(
    Output('result-classifier-2', 'value'),
    [Input('predict-button-classifier-2', 'n_clicks')],
    [State('post-classifier-1', 'value')])
def predict_classifier_2(n_clicks, post):
    #vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
    #                      min_df=1, max_df=1, strip_accents='unicode', use_idf=1,
    #                      smooth_idf=1, sublinear_tf=1)
    result = []
    if n_clicks > 0:
        df_post = DataFrame(columns=["text"])
        df_post["text"] = [clean_text(post.lower())]
        post_prepared = vec_integratedml_bin.transform(df_post["text"])
        sp_matrix_x_train = pd.DataFrame.sparse.from_spmatrix(post_prepared)
        columns = []
        vals = list(sp_matrix_x_train.values[0])
        for x in range(900):
            columns.append("c"+str(x)+"="+str(vals[x]))
        str_columns = ", ".join(columns)
        curs = conn.cursor()
        for tag in predict_tags:
            transformed_tag = "tag_" + str(re.subn(r"[\é\s\\\(\)\.\,\$\&\+\/\?\%\|\"\#\-]", "_", tag)[0])
            if get_integratedml_prediction(transformed_tag, str_columns, curs) == 1:
                result.append(tag)

    return result


@app.callback(
    Output('post-classifier-1', 'value'),
    [Input('load-url-button', 'n_clicks')],
    [State('txt_url', 'value')])
def load_url_into_text(n_clicks, url):
    result = ""
    if n_clicks > 0:
        page = urllib.request.urlopen(url)
        soup = BeautifulSoup(page)
        title = soup.head.title.string
        text = soup.find_all("div", class_="field-item")[0].get_text()
        result = title + " " + text
    return result



navbar = dbc.NavbarSimple(id="list_menu_content", children=[
    dbc.NavItem(dbc.NavLink("Post Tag Classifier 1", href="/post-tag-classifier-1")),
    dbc.NavItem(dbc.NavLink("Vote in iris-ml-suite!",
                            href="https://openexchange.intersystems.com/contest/current", target="_blank")
                )])



# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname, suppress_callback_exceptions=True):
    if pathname == '/post-tag-classifier-1':
        return get_post_tag_classifier_1()
    else:
        return get_post_tag_classifier_1()


if __name__ == '__main__':
    app.layout = html.Div([dcc.Location(id='url', refresh=False),
                        html.Div(navbar),
                        html.Div(id='page-content', style={"margin-left":"15px",
                                                           "margin-right":"15px"})])
    app.run_server(debug=True,host='0.0.0.0')