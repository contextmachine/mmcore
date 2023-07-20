# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |

FROM python:latest


# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла
# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
#COPY --chown=root:root env.yaml /tmp/env.yaml

#RUN micromamba install -y -n base -f /tmp/env.yaml && \
#    micromamba clean --all --yes
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
# ENV PARAM=value

RUN apt update && sudo install libpython3-dev


RUN /bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"

WORKDIR /mmcore
COPY --link . .
#RUN apt update && apt -y install npm nodejs
EXPOSE 7711


RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install .

#ENTRYPOINT ["python3", "-m", "mmcore.serve", "--serve-start=true"]
