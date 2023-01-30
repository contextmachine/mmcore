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
RUN conda env update -f environment.yml && conda init --all
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV PYTHONPATH=${CONDA_DIR}/bin/python
RUN ${CONDA_DIR}/bin/python mmcore/bin/occ_resolver.py && ${CONDA_DIR}/bin/python -m pip install .
# Чтобы следующая команджа работала правильно включите в команду сборки следующее:
#   `docker buildx build --secret id=aws,src=$HOME/.aws/credentials .`
ENTRYPOINT ["${CONDA_DIR}/bin/python"]