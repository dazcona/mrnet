echo 'These commands have to be run outside the docker container!'

echo 'Tag my docker image'
docker tag mrnet_image davidazcona/mrnet:1

echo 'Push it to Dockerhub'
docker push davidazcona/mrnet:1