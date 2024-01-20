docker build  --platform linux/amd64 --tag mmcore_props_update:test .

docker run --platform linux/amd64 --name props-upate2 --rm -it --tty -p 0.0.0.0:7712:7711  mmcore_props_update:test . python3 boxes.py
