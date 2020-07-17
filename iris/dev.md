# useful commands
## build container with no cache
```
docker-compose build --no-cache
```
## open terminal to docker
```
docker-compose exec iris iris session iris -U IRISAPP
```
## export IRIS Analytics artifacts
```
d ##class(dev.code).export("*.DFI")
```

## clean up docker 
```
docker system prune -f
```

docker run --name banzai-contest -d --publish 9091:51773 --publish 9092:52773 store/intersystems/iris-aa-community:2020.3.0AA.331.0
docker exec -it iris_iris_1 bash
docker exec -it iris_iris_1 iris session iris




set filename = "/irisdev/app/csv_y_test.csv" 
set pclass = "community.ytest"
do ##class(community.csvgen).Generate(filename,,.pclass)

DO $SYSTEM.SQL.Shell()

ssh -i /Users/macbook/iris-ml-suite/server/ssh_keys AzureUser@52.142.62.199


docker exec -it iris_iris-ml_1 iris session iris