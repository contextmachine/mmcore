FROM condaforge/mambaforge
USER root
WORKDIR /mmcore
# workspace:pridex:
# workspace:internal
COPY environment.yml environment.yml
RUN conda update -n base -c conda-forge conda
RUN conda env create --file environment.yml && conda init --all
COPY . .
RUN python -m pip install -e .
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV REDIS_DB=0 REDIS_STATESTREAM_ID=0 REDIS_STATESTREAM_KEY=tests:ug-stream REDIS_URL=redis://localhost:6380 PWD=/app

VOLUME ["/mmcore/data"]