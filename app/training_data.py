import jaydebeapi
from pandas import DataFrame

jdbc_server = "jdbc:IRIS://localhost:51773/PYTHON"
jdbc_driver = 'com.intersystems.jdbc.IRISDriver'
iris_jdbc_jar = "./intersystems-jdbc-3.1.0.jar"
iris_user = "_SYSTEM"
iris_password = "SYS"


#getting all tags expected
conn = jaydebeapi.connect(jdbc_driver, jdbc_server, [iris_user, iris_password], iris_jdbc_jar)
curs = conn.cursor()
curs.execute("SELECT ID FROM Community.Tag")
iris_tags = curs.fetchall()
df_iris_tags = DataFrame(iris_tags)
conn.close()


#creating an start point for view

view_iris = []
cmd = "SELECT top 20 id, Name, Tags, Text, PostType FROM Community.Post Where lang = 'en' and tags like '%{}%'"
for x in df_iris_tags[[0]].values:
    view_iris.append(cmd.format(x[0]))



print("ok")