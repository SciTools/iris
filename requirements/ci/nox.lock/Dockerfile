FROM condaforge/mambaforge:latest

LABEL author="scitools" \
      version="0.1" \
      description="SciTools Cirrus-CI mambaforge image"

# python version in "<major><minor>" format e.g., "38"
ARG PY_VER

ENV WORK_DIR="/root/scitools" \
    LOCK_FILE="/root/scitools/iris.lock" \
    NOX_DIR="/root/scitools/.nox" \
    ENV_DIR="/root/scitools/.nox/tests" \
    TMP_DIR="/root/scitools/.nox/tests/tmp"

# configure conda and install conda-lock et al
RUN conda config --set always_yes yes \
  && conda config --set changeps1 no \
  && conda config --set show_channel_urls True \
  && conda config --add channels conda-forge \
  && conda update --quiet --name base conda \
  && mamba install --yes --quiet --channel conda-forge --name base conda-lock nox pip setuptools wheel

# populate work directory with lock file from docker context
WORKDIR ${WORK_DIR}
COPY ./py${PY_VER}-linux-64.lock ${LOCK_FILE}

# install the lock packages into the nox workspace and
# configure the required nox environments
RUN mkdir -p ${ENV_DIR} \
  && mamba install --prefix ${ENV_DIR} --file ${LOCK_FILE} \
  && ln -s ${ENV_DIR} ${NOX_DIR}/gallery \
  && ln -s ${ENV_DIR} ${NOX_DIR}/doctest \
  && ln -s ${ENV_DIR} ${NOX_DIR}/linkcheck \
  && mkdir -p ${TMP_DIR} \
  && sha256sum ${LOCK_FILE} | cut -d' ' -f1 > ${TMP_DIR}/py${PY_VER}.yml
