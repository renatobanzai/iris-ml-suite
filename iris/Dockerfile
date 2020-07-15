ARG IMAGE=store/intersystems/iris-community:2020.1.0.204.0
ARG IMAGE=intersystemsdc/iris-community:2020.1.0.209.0-zpm
ARG IMAGE=intersystemsdc/iris-community:2020.2.0.196.0-zpm
ARG IMAGE=intersystemscommunity/mlte:latest
ARG IMAGE=intersystemscommunity/mlte:aa
FROM $IMAGE

USER root

WORKDIR /opt/irisapp
RUN chown ${ISC_PACKAGE_MGRUSER}:${ISC_PACKAGE_IRISGROUP} /opt/irisapp
COPY irissession.sh /
RUN chmod +x /irissession.sh 

USER ${ISC_PACKAGE_MGRUSER}

COPY  *.csv ./
COPY  Installer.cls .
COPY  src src
SHELL ["/irissession.sh"]

RUN \
  do $SYSTEM.OBJ.Load("Installer.cls", "ck") \
  set sc = ##class(App.Installer).setup() \
  zn "PYTHON" \
  zpm "install webterminal" \
  zpm "install csvgen" \
  set fytrain = "ytrain.csv" \
  set fxtrain = "xtrain.csv" \
  set fytest = "ytest.csv" \
  set fxtest = "xtest.csv" \
  set pclassytrain = "community.ytrain" \
  set pclassxtrain = "community.xtrain" \
  set pclassytest = "community.ytest" \
  set pclassxtest = "community.xtest" \
  do ##class(community.csvgen).Generate(fytrain,,.pclassytrain) \
  do ##class(community.csvgen).Generate(fxtrain,,.pclassxtrain) \
  do ##class(community.csvgen).Generate(fytest,,.pclassytest) \
  do ##class(community.csvgen).Generate(fxtest,,.pclassxtest)

# bringing the standard shell back
SHELL ["/bin/bash", "-c"]
