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
import pandas as pd


jdbc_server = "jdbc:IRIS://20.185.90.39:51773/PYTHON"
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
                   "and not Description is null "
                   "order by id")
total_vocab = curs_vocab.fetchall()
df_vocab = DataFrame(columns=["vocab"], data=total_vocab)
df_vocab = df_vocab.applymap(lambda s: s.lower() if type(s) == str else s)

curs_tags = conn.cursor()
curs_tags.execute("SELECT  ID "
                   "FROM Community.Tag "
                   "where not id is null order by id")
total_tags = curs_tags.fetchall()
df_tags = DataFrame(columns=["tags"], data=total_tags)
df_tags = df_tags.applymap(lambda s: s.lower() if type(s) == str else s)


df = DataFrame(total_cache)
df.columns = [x[0].lower() for x in curs.description]

def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"c\+\+", "cplusplus ", text)
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


df = prepare_dataframe(df)
all_tags = get_all_tags(df[["tags"]])

mlb = MultiLabelBinarizer(classes=all_tags)
y_total = mlb.fit_transform(df["tags"])

n = df.shape[0]

vec = CountVectorizer(ngram_range=(1,1), strip_accents='unicode',
                      max_features=900,stop_words=stop_words)

vec.fit(df_vocab["vocab"])

percent_training = 0.8
line = int(percent_training * n)

df_x_train = df["text"][:line]
df_x_test = df["text"][line:]

x_train = vec.transform(df_x_train)

#saving a pickle with the vectorizer model
filename = 'vec_integratedml.sav'
pickle.dump(vec, open(filename, 'wb'))

x_test = vec.transform(df_x_test)


# creating data frames to create a csv to send to intersystems iris
sp_matrix_x_train = pd.DataFrame.sparse.from_spmatrix(x_train)
sp_matrix_x_test = pd.DataFrame.sparse.from_spmatrix(x_test)

# adding a c prefix to create columns with alphanumeric names
sp_matrix_x_test.columns = ["c" + str(c) for c in sp_matrix_x_test.columns]
sp_matrix_x_train.columns = ["c" + str(c) for c in sp_matrix_x_train.columns]

sp_matrix_x_test.to_csv("xtest.csv", index=False)
sp_matrix_x_train.to_csv("xtrain.csv", index=False)

# formating names to be usable in intersystems iris
formated_columns = ["tag_" + re.subn(r"[\é\s\\\(\)\.\,\$\&\+\/\?\%\|\"\#\-]", "_", x.strip())[0] for x in mlb.classes_]


with open('formated_columns.json', 'w') as outfile:
    json.dump(formated_columns, outfile)

#x_train = x_total[:line]
y_train = DataFrame(y_total[:line])

#x_test = x_total[line:]
y_test = DataFrame(y_total[line:])

y_test.columns = mlb.classes_
y_train.columns = mlb.classes_

csv_y_train = y_train.copy()
csv_y_test = y_test.copy()

csv_y_test.columns = formated_columns
csv_y_train.columns = formated_columns

csv_y_test.to_csv("ytest.csv", index=False)
csv_y_train.to_csv("ytrain.csv", index=False)

#creating 2 dictionaries to convert and revert the names
iris_columns = {}
python_columns = {}
all_views = []
all_models = []
all_trains = []


curs_loop = conn.cursor()
for i, x in enumerate(formated_columns):
    python_columns[x]=mlb.classes_[i]
    iris_columns[mlb.classes_[i]]=x

    #creating sql objects to perform the training
    drop_view_text = "DROP VIEW community.view_train_{}".format(x)
    view_text = "CREATE VIEW community.view_train_{} " \
                "AS " \
                "SELECT " \
                "c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, c67, c68, c69, c70, c71, c72, c73, c74, c75, c76, c77, c78, c79, c80, c81, c82, c83, c84, c85, c86, c87, c88, c89, c90, c91, c92, c93, c94, c95, c96, c97, c98, c99, c100, c101, c102, c103, c104, c105, c106, c107, c108, c109, c110, c111, c112, c113, c114, c115, c116, c117, c118, c119, c120, c121, c122, c123, c124, c125, c126, c127, c128, c129, c130, c131, c132, c133, c134, c135, c136, c137, c138, c139, c140, c141, c142, c143, c144, c145, c146, c147, c148, c149, c150, c151, c152, c153, c154, c155, c156, c157, c158, c159, c160, c161, c162, c163, c164, c165, c166, c167, c168, c169, c170, c171, c172, c173, c174, c175, c176, c177, c178, c179, c180, c181, c182, c183, c184, c185, c186, c187, c188, c189, c190, c191, c192, c193, c194, c195, c196, c197, c198, c199, c200, c201, c202, c203, c204, c205, c206, c207, c208, c209, c210, c211, c212, c213, c214, c215, c216, c217, c218, c219, c220, c221, c222, c223, c224, c225, c226, c227, c228, c229, c230, c231, c232, c233, c234, c235, c236, c237, c238, c239, c240, c241, c242, c243, c244, c245, c246, c247, c248, c249, c250, c251, c252, c253, c254, c255, c256, c257, c258, c259, c260, c261, c262, c263, c264, c265, c266, c267, c268, c269, c270, c271, c272, c273, c274, c275, c276, c277, c278, c279, c280, c281, c282, c283, c284, c285, c286, c287, c288, c289, c290, c291, c292, c293, c294, c295, c296, c297, c298, c299, c300, c301, c302, c303, c304, c305, c306, c307, c308, c309, c310, c311, c312, c313, c314, c315, c316, c317, c318, c319, c320, c321, c322, c323, c324, c325, c326, c327, c328, c329, c330, c331, c332, c333, c334, c335, c336, c337, c338, c339, c340, c341, c342, c343, c344, c345, c346, c347, c348, c349, c350, c351, c352, c353, c354, c355, c356, c357, c358, c359, c360, c361, c362, c363, c364, c365, c366, c367, c368, c369, c370, c371, c372, c373, c374, c375, c376, c377, c378, c379, c380, c381, c382, c383, c384, c385, c386, c387, c388, c389, c390, c391, c392, c393, c394, c395, c396, c397, c398, c399, c400, c401, c402, c403, c404, c405, c406, c407, c408, c409, c410, c411, c412, c413, c414, c415, c416, c417, c418, c419, c420, c421, c422, c423, c424, c425, c426, c427, c428, c429, c430, c431, c432, c433, c434, c435, c436, c437, c438, c439, c440, c441, c442, c443, c444, c445, c446, c447, c448, c449, c450, c451, c452, c453, c454, c455, c456, c457, c458, c459, c460, c461, c462, c463, c464, c465, c466, c467, c468, c469, c470, c471, c472, c473, c474, c475, c476, c477, c478, c479, c480, c481, c482, c483, c484, c485, c486, c487, c488, c489, c490, c491, c492, c493, c494, c495, c496, c497, c498, c499, c500, c501, c502, c503, c504, c505, c506, c507, c508, c509, c510, c511, c512, c513, c514, c515, c516, c517, c518, c519, c520, c521, c522, c523, c524, c525, c526, c527, c528, c529, c530, c531, c532, c533, c534, c535, c536, c537, c538, c539, c540, c541, c542, c543, c544, c545, c546, c547, c548, c549, c550, c551, c552, c553, c554, c555, c556, c557, c558, c559, c560, c561, c562, c563, c564, c565, c566, c567, c568, c569, c570, c571, c572, c573, c574, c575, c576, c577, c578, c579, c580, c581, c582, c583, c584, c585, c586, c587, c588, c589, c590, c591, c592, c593, c594, c595, c596, c597, c598, c599, c600, c601, c602, c603, c604, c605, c606, c607, c608, c609, c610, c611, c612, c613, c614, c615, c616, c617, c618, c619, c620, c621, c622, c623, c624, c625, c626, c627, c628, c629, c630, c631, c632, c633, c634, c635, c636, c637, c638, c639, c640, c641, c642, c643, c644, c645, c646, c647, c648, c649, c650, c651, c652, c653, c654, c655, c656, c657, c658, c659, c660, c661, c662, c663, c664, c665, c666, c667, c668, c669, c670, c671, c672, c673, c674, c675, c676, c677, c678, c679, c680, c681, c682, c683, c684, c685, c686, c687, c688, c689, c690, c691, c692, c693, c694, c695, c696, c697, c698, c699, c700, c701, c702, c703, c704, c705, c706, c707, c708, c709, c710, c711, c712, c713, c714, c715, c716, c717, c718, c719, c720, c721, c722, c723, c724, c725, c726, c727, c728, c729, c730, c731, c732, c733, c734, c735, c736, c737, c738, c739, c740, c741, c742, c743, c744, c745, c746, c747, c748, c749, c750, c751, c752, c753, c754, c755, c756, c757, c758, c759, c760, c761, c762, c763, c764, c765, c766, c767, c768, c769, c770, c771, c772, c773, c774, c775, c776, c777, c778, c779, c780, c781, c782, c783, c784, c785, c786, c787, c788, c789, c790, c791, c792, c793, c794, c795, c796, c797, c798, c799, c800, c801, c802, c803, c804, c805, c806, c807, c808, c809, c810, c811, c812, c813, c814, c815, c816, c817, c818, c819, c820, c821, c822, c823, c824, c825, c826, c827, c828, c829, c830, c831, c832, c833, c834, c835, c836, c837, c838, c839, c840, c841, c842, c843, c844, c845, c846, c847, c848, c849, c850, c851, c852, c853, c854, c855, c856, c857, c858, c859, c860, c861, c862, c863, c864, c865, c866, c867, c868, c869, c870, c871, c872, c873, c874, c875, c876, c877, c878, c879, c880, c881, c882, c883, c884, c885, c886, c887, c888, c889, c890, c891, c892, c893, c894, c895, c896, c897, c898, c899, " \
                "{} " \
                "FROM " \
                "community.xtrain as xtrain " \
                "inner join " \
                "community.ytrain as ytrain " \
                "ON " \
                "ytrain.id = xtrain.id".format(x,x)

    model_text = "CREATE MODEL has_{}_tag PREDICTING ({}) FROM community.view_train_{}".format(x,x,x)

    train_text = "TRAIN MODEL has_{}_tag FROM community.view_train_{}".format(x,x,x)
    print(x)
    try:
        curs_loop.execute(view_text)
        curs_loop.execute(model_text)
        curs_loop.execute(train_text)
    except:
        print(x)


    all_views.append("Set tSC = ##class(%SQL.Statement).%ExecDirect(, \"{}\")".format(view_text))
    all_models.append("Set tSC = ##class(%SQL.Statement).%ExecDirect(, \"{}\")".format(model_text))
    all_trains.append("Set tSC = ##class(%SQL.Statement).%ExecDirect(, \"{}\")".format(train_text))

predictors = {}
