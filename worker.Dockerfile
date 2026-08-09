FROM ubuntu:22.04

WORKDIR /workspace

COPY accelergy-timeloop-infrastructure accelergy-timeloop-infrastructure
COPY test_tl test_tl

RUN apt-get update && \
	apt-get install -y scons build-essential -y python3.11 python3.11-venv  \
    libncurses-dev libconfig++-dev libyaml-cpp-dev libboost-serialization-dev libboost-iostreams-dev libgpm-dev

ARG BIN_DEST="/usr/local/bin"
ARG SRC_DIR="accelergy-timeloop-infrastructure/src"
ARG ACCEL_VENV_DEST="/opt/accel_venv"

RUN python3.11 -m venv $ACCEL_VENV_DEST
ENV PATH="$ACCEL_VENV_DEST/bin:$PATH"

RUN pip install wheel && \
    make -C $SRC_DIR/cacti -f makefile -j$(nproc) && \
    cp -r $SRC_DIR/cacti $BIN_DEST && \
    pip3 install $SRC_DIR/accelergy && \
    cp $(which accelergy) $BIN_DEST/accelergy && \
    pip3 install $SRC_DIR/accelergy-aladdin-plug-in && \
    pip3 install accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in && \
    pip3 install $SRC_DIR/accelergy-table-based-plug-ins && \
    ln -s "$(pwd)/$SRC_DIR/timeloop/pat-public/src/pat" $SRC_DIR/timeloop/src/pat && \
    scons -C $SRC_DIR/timeloop -j$(nproc) --accelergy --static && \
    cp -r $SRC_DIR/timeloop/build/timeloop-* $BIN_DEST

COPY src src
COPY timeloop-accelergy-exercises timeloop-accelergy-exercises
ENV PATH="$.venv/bin:$PATH"
RUN python3.11 -m venv .venv && \
    pip3 install pydantic && \
    pip3 install pika

ENTRYPOINT ["python", "-m", "src.mapping.impl.timeloop_consumer"]