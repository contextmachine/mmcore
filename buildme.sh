DOCKER_BUILDKIT=1 docker build --no-cache --platform amd64 -t sthv/mmcore:amd64 . && docker run --name mmcore -t -i -e ./.env -w /mmcore -P --privileged sthv/mmcore:amd64
