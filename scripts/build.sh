#!/usr/bin/env bash
set -euxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_DIR=$SCRIPT_DIR/..
CXX_DIR=$PROJECT_DIR/cxx
BUILD_DIR=$CXX_DIR/build

echo "Ensuring $PROJECT_DIR has libcxxwrap_julia_jll in dev mode"
julia \
  --project=$PROJECT_DIR \
  -e "using Pkg; Pkg.develop(\"libcxxwrap_julia_jll\")"

echo "Patching $CXX_DIR/libfranka"
cd $CXX_DIR/libfranka
patch -p1 < $CXX_DIR/libfranka-0001-Add-missing-include-s.patch

echo "Building $CXX_DIR"
cmake \
  -B $BUILD_DIR \
  -S $CXX_DIR \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DJlCxx_DIR=$(julia --project=$PROJECT_DIR -e "using CxxWrap; println(CxxWrap.prefix_path())") \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build $BUILD_DIR
ln -sf $BUILD_DIR/compile_commands.json $PROJECT_DIR/compile_commands.json
echo "$BUILD_DIR successfully built"

echo "Unpatching $CXX_DIR/libfranka"
cd $CXX_DIR/libfranka
git reset --hard HEAD
