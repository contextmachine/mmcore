FROM condaforge/mambaforge
USER root
WORKDIR /app

# workspace:pridex:
# workspace:internal
COPY environment.yml environment.yml
RUN conda update -n base -c conda-forge conda
RUN conda env create --file environment.yml && conda init --all
COPY . .
ENTRYPOINT ["python app.py"]
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
