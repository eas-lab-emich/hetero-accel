#!/bin/bash

VENV_NAME=".venv"

git submodule update --init --recursive

create_directory () {
  pending_dir=$1
  if [ ! -d "$pending_dir" ]; then
    echo "$pending_dir does not exist."
    mkdir "$pending_dir"
    echo "$pending_dir created."
  else
    echo "$pending_dir exists"
  fi
}

local_directory="$HOME/.local"
create_directory "$local_directory"
create_directory "$local_directory/bin"
create_directory "$local_directory/share"
mkdir -p "$HOME/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/"


# adding dirs to PATH as necessary
DIRS=("$HOME/.local/bin" "$HOME/.local/share")

for TARGET_DIR in "${DIRS[@]}"; do
    # Check if directory is already in PATH
    if [[ ":$PATH:" == *":$TARGET_DIR:"* ]]; then
        echo "$TARGET_DIR already in PATH"
    else
        echo "Adding $TARGET_DIR to PATH in ~/.bashrc"

        # Only add if not already in bashrc
        if ! grep -qF "$TARGET_DIR" "$HOME/.bashrc"; then
            echo "export PATH=\"\$PATH:$TARGET_DIR\"" >> "$HOME/.bashrc"
            export PATH=\"\$PATH:$TARGET_DIR\"
        else
            echo " - Skipped: $TARGET_DIR already present in ~/.bashrc"
        fi
    fi
done

# install bazel build system
if [ ! -f "$local_directory/bin/bazel" ]; then
  curl -L -o "$HOME/.local/bin/bazel" "https://github.com/bazelbuild/bazel/releases/download/7.7.1/bazel-7.7.1-linux-x86_64"
  chmod 755 "$HOME/.local/bin/bazel"
fi

# create venv, install project requirements
if [ ! -d "$VENV_NAME" ]; then
    echo "$VENV_NAME venv does not exist."
    python3 -m venv .venv
    if [ $? -eq 0 ]; then
      echo "Created $VENV_NAME successfully"
    else
      echo "Failed to create venv $VENV_NAME. Ending script early..."
      return 1
    fi
else
  echo "$VENV_NAME venv already exists, continuing."
fi

source $VENV_NAME/bin/activate
pip3 install -r setup/requirements.txt

make -C accelergy-timeloop-infrastructure/src/cacti -f makefile
pip3 install accelergy-timeloop-infrastructure/src/accelergy
pip3 install accelergy-timeloop-infrastructure/src/accelergy-aladdin-plug-in
pip3 install accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in
cp -r accelergy-timeloop-infrastructure/src/cacti "$HOME/.local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/"
pip3 install accelergy-timeloop-infrastructure/src/accelergy-table-based-plug-ins
ln -s "$(pwd)/accelergy-timeloop-infrastructure/src/timeloop/pat-public/src/pat" accelergy-timeloop-infrastructure/src/timeloop/src/pat
scons -C accelergy-timeloop-infrastructure/src/timeloop -j32 --accelergy --static
cp -r accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-* ~/.local/bin
cd generalizedassignmentsolver || { echo "Failed to enter generalizedassignmentsolver directory, aborting"; return 1; }
bazel build -- ...
cd ..

imagenet_dir=../data/Imagenet
if [ ! -d $imagenet_dir ]; then
    echo "$imagenet_dir does not exist."
    mkdir -p $imagenet_dir
    if [ $? -eq 0 ]; then
      echo "Created $imagenet_dir successfully"
    else
      echo "Failed to create $imagenet_dir. Ending script early..."
      return 1
    fi
    cp -r setup/data/Imagenet/* $imagenet_dir
    ln -s /data/imagenet $imagenet_dir/val
else
  echo "imagenet_dir already exists, continuing."
fi