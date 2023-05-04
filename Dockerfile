# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |

FROM mambaorg/micromamba
USER root
# Для выполнения директивы ниже вам необходимо указать `syntax=docker/dockerfile:1` в начале файла
# 🐍 Setup micromamba.
# ⚙️ Source: https://hub.docker.com/r/mambaorg/micromamba
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
# ENV PARAM=value
WORKDIR /mmcore
COPY --link . .
RUN rm -r /root/.cache
RUN apt update && apt -y install npm nodejs
RUN /usr/local/bin/_entrypoint.sh python bin/platform_resolver.py
RUN /usr/local/bin/_entrypoint.sh python -m pip install -e . --no-deps
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python"]