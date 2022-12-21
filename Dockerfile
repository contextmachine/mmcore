FROM continuumio/anaconda-pkg-build
MAINTAINER "CONTEXTMACHINE"
USER root
WORKDIR cxm
COPY environment.yml /cxm/environment.yml
RUN conda env update -n "base" --file=environment.yml
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
COPY mmcore /party/mmcore
COPY setup.py /party/mmcore
RUN /bin/bash conda activate && python -m pip install pip --upgrade && python -m pip install ./mmcore
# 🐳 Setting pre-build params and environment variables.
# ⚙️ Please set you environment globals :)
ENV MY_FAVORIT_TRANSPORT="CONTEXTMACHINE"
ENTRYPOINT ["/bin/bash"]