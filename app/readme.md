#starting the server

docker run --name banzai-contest -d --publish 9091:51773 --publish 9092:52773 store/intersystems/iris-aa-community:2020.3.0AA.331.0

docker exec -it iris_iris_1 bash
docker exec -it iris_iris_1 iris session iris

