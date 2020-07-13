import jaydebeapi
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import re, string
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


jdbc_server = "jdbc:IRIS://localhost:51773/PYTHON"
jdbc_driver = 'com.intersystems.jdbc.IRISDriver'
iris_jdbc_jar = "./intersystems-jdbc-3.1.0.jar"
iris_user = "_SYSTEM"
iris_password = "SYS"

conn = jaydebeapi.connect(jdbc_driver, jdbc_server, [iris_user, iris_password], iris_jdbc_jar)
curs = conn.cursor()
curs.execute("SELECT "
             " id, Name, Tags, Text "
             "FROM Community.Post_Train  "
             "Where  "
             "not text is null "             
             "order by id")

total_cache = curs.fetchall()

#getting all description of each tag to compose the vocabulary
curs_vocab = conn.cursor()
curs_vocab.execute("SELECT  ID||' '||Description "
                   "FROM Community.Tag "
                   "where not id is null "
                   "and not Description is null")
total_vocab = curs_vocab.fetchall()
df_vocab = DataFrame(columns=["vocab"], data=total_vocab)
df_vocab = df_vocab.applymap(lambda s: s.lower() if type(s) == str else s)

curs_tags = conn.cursor()
curs_tags.execute("SELECT  ID "
                   "FROM Community.Tag "
                   "where not id is null ")
total_tags = curs_tags.fetchall()
df_tags = DataFrame(columns=["tags"], data=total_tags)
df_tags = df_tags.applymap(lambda s: s.lower() if type(s) == str else s)


df = DataFrame(total_cache)
df.columns = [x[0].lower() for x in curs.description]

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

def prepare_dataframe(_df):
    #converting all to lower case
    _df = _df.applymap(lambda s: s.lower() if type(s) == str else s)
    _df["tags"] = tuple(_df["tags"].str.split(","))
    _df["text"] = _df["text"].map(lambda com : clean_text(com))
    return _df


def get_all_tags(tags_list):
    real_tags = df_tags["tags"].values.tolist()
    all_tags = []
    for x in tags_list.values:
        all_tags += x[0]

    result = list(set(all_tags))
    result = [x for x in result if x in real_tags]

    #result.remove("article")
    #result.remove("question")
    #result.remove("caché")
    with open('all_tags.json', 'w') as outfile:
        json.dump(result, outfile)
    return tuple(set(result))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

df = prepare_dataframe(df)
all_tags = get_all_tags(df[["tags"]])

mlb = MultiLabelBinarizer(classes=all_tags)
y_total = mlb.fit_transform(df["tags"])

n = df.shape[0]
#vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize, strip_accents='unicode', use_idf=1,
#                      smooth_idf=1, sublinear_tf=1, stop_words=stop_words)

vec = CountVectorizer(ngram_range=(1,1), tokenizer=tokenize, strip_accents='unicode',
                      stop_words=stop_words)

#x_total = vec.fit_transform(df["text"])
#filename = 'vec.sav'
#pickle.dump(vec, open(filename, 'wb'))

percent_training = 0.8
line = int(percent_training * n)

df_x_train = df["text"][:line]
df_x_test = df["text"][line:]


vec.fit(df_vocab["vocab"])

x_train = vec.transform(df_x_train)

#saving a pickle with the vectorizer model
filename = 'vec.sav'
pickle.dump(vec, open(filename, 'wb'))

x_test = vec.transform(df_x_test)

#x_train = x_total[:line]
y_train = DataFrame(y_total[:line])

#x_test = x_total[line:]
y_test = DataFrame(y_total[line:])

y_test.columns = mlb.classes_
y_train.columns = mlb.classes_

predictors = {}
for tag in all_tags:
    print('**Processing {} posts...**'.format(tag))

    # Training logistic regression model on train data
    predictors[tag] = Pipeline([('clf', OneVsRestClassifier(
        LogisticRegression(solver='sag', max_iter=4000), n_jobs=-1)),]
                               ).fit(x_train, y_train[tag])

    #predictors[tag] = Pipeline([('clf', OneVsRestClassifier(
    #    RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=200), n_jobs=-1)),]
    #                           ).fit(x_train, y_train[tag])

    #predictors[tag] = Pipeline([('clf', OneVsRestClassifier(
    #    SVC(kernel="rbf", random_state=1, C=5)
    #    , n_jobs=-1)),]
    #                          ).fit(x_train, y_train[tag])



    # calculating test accuracy
    prediction = predictors[tag].predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(y_test[tag], prediction)))
    print("\n")

filename = 'predictors.sav'
pickle.dump(predictors, open(filename, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)

print("ok")


