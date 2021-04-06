#!/bin/bash
set -ex

mkdir -p $HOME/opt/openmc_dependencies

# Download HDF5 data
if [[ ! -e $HOME/opt/openmc_dependencies/nndc_hdf5/cross_sections.xml ]]; then
    wget -q -O - https://anl.box.com/shared/static/teaup95cqv8s9nn56hfn7ku8mmelr95p.xz | tar -C $HOME/opt/openmc_dependencies -xJ
fi

# Download ENDF/B-VII.1 distribution
ENDF=$HOME/opt/openmc_dependencies/endf-b-vii.1
if [[ ! -d $ENDF/neutrons || ! -d $ENDF/photoat || ! -d $ENDF/atomic_relax ]]; then
    wget -q -O - https://anl.box.com/shared/static/4kd2gxnf4gtk4w1c8eua5fsua22kvgjb.xz | tar -C $HOME/opt/openmc_dependencies -xJ
fi
