# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |


FROM buildpack-deps as deps
LABEL org.opencontainers.image.source=https://github.com/contextmachine/mmcore
LABEL org.opencontainers.image.description="mmcore"
LABEL autor="Andrew Astakhov <aa@contextmachine.ru> <aw.astakhov@gmail.com>"
# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла
# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
#COPY --chown=root:root env.yaml /tmp/env.yaml

#RUN micromamba install -y -n base -f /tmp/env.yaml && \
#    micromamba clean --all --yes
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)


RUN apt update -y && apt install python3.11-full -y && apt install python3-pip -y

FROM deps

WORKDIR /mmcore
COPY --link . .
#RUN apt update && apt -y install npm nodejs
EXPOSE 7711

RUN python3.11 -m pip install -e . --break-system-packages


#ENTRYPOINT ["python3", "-m", "mmcore.serve", "--serve-start=true"]
