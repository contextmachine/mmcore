DOCKER_BUILDKIT=1 docker build --no-cache --platform amd64 -t sthv/mmcore:amd64 . && docker run --rm --name mmcore -t -i -e ./.env -w /mmcore -P --privileged sthv/mmcore:amd64
docker tag sthv/mmcore:amd64 cr.yandex/crpfskvn79g5ht8njq0k/mmcore:amd64

