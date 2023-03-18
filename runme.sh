docker run \
  -p 0.0.0.0:7777:7777 \
  -p 0.0.0.0:6666:6666/udp \
  -p 0.0.0.0:8777:8080 \
  -p 0.0.0.0:8666:8080/udp \
  -p 0.0.0.0:9777:8000 \
  -p 0.0.0.0:9666:8000/udp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /Users/andrewastakhov/PycharmProjects/:/mm \
  --rm \
  --name mmcore \
  --privileged \
  sthv/mmcore:amd64 \
  python -c "import os,sys;import mmcore;print(mmcore.__version__());sys.exit()"