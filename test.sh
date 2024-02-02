docker run --rm --tty --name mmcore2 ghcr.io/contextmachine/mmcore:latest &
docker cp tests/dockertest.py mmcore2:/mmcore/dockertest.py
docker exec mmcore python3 dockertest.py
