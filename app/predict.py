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
import re, string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# load the model from disk
filename = 'predictors.sav'
predictors = pickle.load(open(filename, 'rb'))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

filename = 'vec.sav'
vec = pickle.load(open(filename, 'rb'))


# load tags
with open('all_tags.json') as json_file:
    all_tags = set(json.load(json_file))



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
            html.Label('Type your post here to predict tags'),
            dcc.Textarea(id='post-classifier-1',
                         style={'width': '100%', 'height': 200}),
            html.Button('Predict', id='predict-button-classifier-1', n_clicks=0),
            html.Div(children=[
                html.Label('Tags'),
                dcc.Dropdown(id='result-classifier-1',
                             multi=True,
                             options=[{"label": opt, "value":opt} for opt in all_tags]
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


navbar = dbc.NavbarSimple(id="list_menu_content", children=[
    dbc.NavItem(dbc.NavLink("Post Tag Classifier 1", href="/post-tag-classifier-1")),
    dbc.NavItem(dbc.NavLink("Post Tag Classifier 2", href="/post-tag-classifier-2")),
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
                        html.Div(id='page-content')])
    app.run_server(debug=True,host='0.0.0.0')