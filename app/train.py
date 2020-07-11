import jaydebeapi
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import re, string
import pickle
import json



jdbc_server = "jdbc:IRIS://localhost:51773/PYTHON"
jdbc_driver = 'com.intersystems.jdbc.IRISDriver'
iris_jdbc_jar = "./intersystems-jdbc-3.1.0.jar"
iris_user = "_SYSTEM"
iris_password = "SYS"

conn = jaydebeapi.connect(jdbc_driver, jdbc_server, [iris_user, iris_password], iris_jdbc_jar)
curs = conn.cursor()
curs.execute("SELECT "
             "top 1000 id, Name, Tags, Text "
             "FROM Community.Post  "
             "Where lang = 'en' "
             "and not trim(isnull(text, ''))='' "
             "and posttype='article'  "
             "order by id")
total_cache = curs.fetchall()

df = DataFrame(total_cache)
df.columns = [x[0].lower() for x in curs.description]

def prepare_dataframe(_df):
    #converting all to lower case
    _df = _df.applymap(lambda s: s.lower() if type(s) == str else s)
    _df["tags"] = tuple(_df["tags"].str.split(","))
    return _df


def get_all_tags(tags_list):
    all_tags = []
    for x in tags_list.values:
        all_tags += x[0]
    with open('all_tags.json', 'w') as outfile:
        json.dump(all_tags, outfile)
    return tuple(set(all_tags))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

df = prepare_dataframe(df)
all_tags = get_all_tags(df[["tags"]])


mlb = MultiLabelBinarizer(classes=all_tags)
y_total = mlb.fit_transform(df["tags"])

n = df.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1 )

x_total = vec.fit_transform(df["text"])

percent_training = 0.8
line = int(percent_training * n)

x_train = x_total[:line]
y_train = DataFrame(y_total[:line])

x_test = x_total[line:]
y_test = DataFrame(y_total[line:])

y_test.columns = mlb.classes_
y_train.columns = mlb.classes_

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=800), n_jobs=-1)),])

predictors = {}

for tag in all_tags:
    print('**Processing {} posts...**'.format(tag))

    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, y_train[tag])
    predictors[tag] = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=800), n_jobs=-1)),])\
        .fit(x_train, y_train[tag])
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(y_test[tag], prediction)))
    print("\n")

filename = 'contest.sav'
pickle.dump(predictors, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

print("ok")


