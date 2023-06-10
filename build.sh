DOCKER_BUILDKIT=1 docker build --platform amd64 -t sthv/mmcore:latest  .
docker tag sthv/mmcore:latest\
           cr.yandex/crpfskvn79g5ht8njq0k/mmcore:latest
