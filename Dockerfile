# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |

FROM python:3.10.11-buster


# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла
# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
#COPY --chown=root:root env.yaml /tmp/env.yaml

#RUN micromamba install -y -n base -f /tmp/env.yaml && \
#    micromamba clean --all --yes
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
# ENV PARAM=value

WORKDIR /mmcore
COPY --link . .
#RUN apt update && apt -y install npm nodejs
RUN python3 -m pip install . && python3 -m pip install -r requirements.txt
ENTRYPOINT ["python", "-m", "mmcore.serve","--serve-start=true"]