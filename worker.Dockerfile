ARG BIN_DEST="/usr/local/bin"
ARG SRC_DIR="accelergy-timeloop-infrastructure/src"
ARG ACCEL_VENV_DEST="/opt/accel_venv"
ARG WORK_DIR=/workspace

FROM ubuntu:22.04 AS build

ARG BIN_DEST
ARG SRC_DIR
ARG ACCEL_VENV_DEST
ARG WORK_DIR

WORKDIR $WORK_DIR

COPY accelergy-timeloop-infrastructure accelergy-timeloop-infrastructure

RUN apt-get update && \
	apt-get install -y scons build-essential python3.11 python3.11-venv  \
    libncurses-dev libconfig++-dev libyaml-cpp-dev libboost-serialization-dev libboost-iostreams-dev libgpm-dev

RUN python3.11 -m venv $ACCEL_VENV_DEST
ENV PATH="$ACCEL_VENV_DEST/bin:$PATH"

RUN pip install wheel && \
    make -C $SRC_DIR/cacti -f makefile -j$(nproc) && \
    pip3 install $SRC_DIR/accelergy && \
    pip3 install $SRC_DIR/accelergy-aladdin-plug-in && \
    pip3 install accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in && \
    pip3 install $SRC_DIR/accelergy-table-based-plug-ins && \
    ln -s $SRC_DIR/timeloop/pat-public/src/pat $SRC_DIR/timeloop/src/pat && \
    scons -C $SRC_DIR/timeloop -j$(nproc) --accelergy --static


FROM ubuntu:22.04

LABEL org.opencontainers.image.title="Hetero-Accel Timeloop Worker"
LABEL org.opencontainers.image.description="Distributed Timeloop/Accelergy mapping worker for Hetero-Accel"
LABEL org.opencontainers.image.source="https://github.com/eas-lab-emich/hetero-accel"
LABEL org.opencontainers.image.version="0.0.4"

ARG BIN_DEST
ARG SRC_DIR
ARG ACCEL_VENV_DEST
ENV ACCEL_VENV_DEST=$ACCEL_VENV_DEST
ARG WORK_DIR

WORKDIR $WORK_DIR

RUN apt-get update && \
	apt-get install -y python3.11 python3.11-venv

COPY --from=build $ACCEL_VENV_DEST $ACCEL_VENV_DEST
COPY --from=build $WORK_DIR/$SRC_DIR/cacti $BIN_DEST/cacti
COPY --from=build $WORK_DIR/$SRC_DIR/timeloop/build/timeloop-* $BIN_DEST

ENV PATH="$WORK_DIR/.venv/bin:$PATH"
COPY timeloop-accelergy-exercises timeloop-accelergy-exercises
COPY setup/accelergy $BIN_DEST/accelergy
RUN python3.11 -m venv .venv && \
  pip3 install pydantic pika pyyaml
COPY src src

ENTRYPOINT ["python", "-m", "src.mapping.impl.timeloop_consumer"]