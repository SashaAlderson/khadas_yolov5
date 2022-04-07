#!/bin/bash

set -e

export OPENCV_ROOT=/usr/include/opencv4
export TENGINE_DIR=/usr
export DDK_DIR=/usr/share/npu/sdk
export TOOLCHAIN=
export CROSS_COMPILE=


make -f makefile-cv4.linux



