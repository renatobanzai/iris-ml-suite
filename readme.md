## Iris ML Suite

In this project I'm going to explain how was my first (and second, and third ++) contact with IRIS IntegratedML and use 
IRIS to compose my Machine Learning Environment.

# Challenges

I decided to take the the suggestion on this post: 
[https://community.intersystems.com/post/posts-and-tags-problem-intersystems-iris-ai-contest](posts-and-tags-problem-intersystems-iris-ai-contest)

Actually two problems. 
1. The author chooses wrong tags.
2. The author chooses no tags.

# Predicting Tags from IRIS Developer Community Posts
![picture](https://raw.githubusercontent.com/renatobanzai/iris-ml-suite/master/img/post_tag_classifier.gif)


### Demo
I have deployed the trained model as a demo here:
[http://iris-ml-suite.eastus.cloudapp.azure.com/](http://iris-ml-suite.eastus.cloudapp.azure.com/)

## First Approach
I start exploring the data to extract some features from the Post dataset. To accelerate I had used the Tags field as 
training set.

## Using the demo

Type or paste a community post on the textarea of site and click in Predict. After this, my model will predict which
 tags should fit better for it. 

## Getting started

### Prerequisites
* git
* docker and docker-compose **adjust docker settings to up memory and cpu the AI demands more capacity**
* access to a terminal in your environment

### Installing
Clone my repository typing these commands:

```
git clone https://github.com/renatobanzai/iris-ml-suite.git
```

### Building and running the docker-compose
**adjust docker settings to up memory and cpu the AI demands more capacity**
- 4GB Memory (or more if you can)
- 2CPU (or more if you can)

### Need to set more memory to docker engine
![picture](https://raw.githubusercontent.com/renatobanzai/iris-python-covid19/master/img/docker_memory.png)

### Running in linux and MacOS
```
docker-compose build

docker-compose up -d
```

### Estimated time to up containers
1st time running will depend of your internet link to download the images and dependencies. 
If it last more than 15 minutes probably something goes wrong feel free to communicate here.
After the 1st time running the next ones will perform better and take less then 5 minutes.


### If is everything ok
After a while you can open your browser and go to the address:

- Main Menu: [http://localhost:8050](http://localhost:8050)

### You should look at IRIS Admin Portal

I'm using for now the PYTHON namespace

```
http://localhost:9092
user: _SYSTEM
pass: SYS
```

