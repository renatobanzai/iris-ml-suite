# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import json

# load the model from disk
filename = 'contest.sav'
predictors = pickle.load(open(filename, 'rb'))

# load tags
with open('all_tags.json') as json_file:
    all_tags = json.load(json_file)

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
    if n_clicks > 0:
        #for tag in all_tags:
        #    predictors[tag].predict(post)
        return ["node.js", "jdbc"]
    else:
        return []


navbar = dbc.NavbarSimple(id="list_menu_content", children=[
    dbc.NavItem(dbc.NavLink("Post Tag Classifier 1", href="/post-tag-classifier-1")),
    dbc.NavItem(dbc.NavLink("Post Tag Classifier 2", href="/post-tag-classifier-2")),
    dbc.NavItem(dbc.NavLink("Vote in iris-python-suite!",
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
    app.run_server(debug=False,host='0.0.0.0')