DOCKER_BUILDKIT=1 docker build --no-cache --platform amd64 -t ghcr.io/contextmachine/mmcore:latest  .
docker tag ghcr.io/contextmachine/mmcore:latest\
           cr.yandex/crpfskvn79g5ht8njq0k/mmcore:latest
