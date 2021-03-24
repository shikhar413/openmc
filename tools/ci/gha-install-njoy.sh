#!/bin/bash
set -ex
if [[ ! -e $HOME/opt/openmc_dependencies/NJOY2016/build/njoy ]]; then
    cd $HOME/opt/openmc_dependencies
    git clone https://github.com/njoy/NJOY2016
    cd NJOY2016
    mkdir build && cd build
    cmake -Dstatic=on .. && make 2>/dev/null
fi
