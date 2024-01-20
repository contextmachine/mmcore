FROM ghcr.io/contextmachine/mmcore:master
LABEL authors="sth-v"
WORKDIR /mmcore
COPY boxes.py boxes.py
EXPOSE 7711
ENV MMCORE_API_PREFIX="/"
ENV MMCORE_ADDRESS="http://localhost:7712"
ENTRYPOINT ["python3", "boxes.py"]