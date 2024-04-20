# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |


FROM buildpack-deps as builder
LABEL org.opencontainers.image.source=https://github.com/contextmachine/mmcore
LABEL org.opencontainers.image.description="mmcore, the modern cad/cam engine"
LABEL autor="Andrew Astakhov <aa@contextmachine.ru> <aw.astakhov@gmail.com>"
# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла
# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
#COPY --chown=root:root env.yaml /tmp/env.yaml

#RUN micromamba install -y -n base -f /tmp/env.yaml && \
#    micromamba clean --all --yes
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
COPY docker-build-step1.sh /docker-build-step1.sh

RUN bash docker-build-step1.sh


FROM builder AS installer
WORKDIR /tmp/build-python
COPY --link docker-build-step2.sh docker-build-step2.sh
RUN bash docker-build-step2.sh

FROM installer
WORKDIR /mmcore
COPY --link . .
#RUN apt update && apt -y install npm nodejs
EXPOSE 7711

RUN python3.12 -m pip install . --break-system-packages


#ENTRYPOINT ["python3.12", "-m", "mmcore.serve", "--serve-start=true"]
