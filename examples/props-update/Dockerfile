FROM ghcr.io/contextmachine/mmcore:master
LABEL authors="sth-v"
WORKDIR /mmcore
COPY . .
EXPOSE 7711
ENV MMCORE_API_PREFIX="/"
ENV MMCORE_ADDRESS="http://localhost:7712"
ENTRYPOINT ["python3", "boxes.py"]