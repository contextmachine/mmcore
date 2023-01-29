# syntax=docker/dockerfile:1

#             //*[@id="understand-how-cmd-and-entrypoint-interact"]
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |


FROM condaforge/mambaforge
# больной ублюдок
USER root
# Нет ну если вы предпочитаете другой стиль можете конечно сделать все чисто ...
#
WORKDIR /mmcore
COPY --link . .
RUN bash mamba env create -f environment.yml && mamba init --all
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV PYTHONPATH=${CONDA_DIR}/envs/mmcore/bin/python
RUN bash ${CONDA_DIR}/envs/mmcore/bin/python -m pip install -e .
# Чтобы следующая команджа работала правильно включите в команду сборки следующее:
#   `docker buildx build --secret id=aws,src=$HOME/.aws/credentials .`
RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
  aws s3 cp s3://storage.yandexcloud.net/lahta.contextmachine.online

ENTRYPOINT ["bash", "${CONDA_DIR}/envs/mmcore/bin/python"]