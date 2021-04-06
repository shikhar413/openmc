
#!/bin/bash
set -ex

ROOT_DIR=$HOME/opt/openmc_dependencies
CURRENT_DIR=$(pwd)

# Eigen Variables
EIGEN_VER=3.3.9

# MOAB Variables
MOAB_BRANCH='Version5.1.0'
MOAB_REPO='https://bitbucket.org/fathomteam/moab/'
MOAB_INSTALL_DIR=$ROOT_DIR/MOAB/

# DAGMC Variables
DAGMC_BRANCH='develop'
DAGMC_REPO='https://github.com/svalinn/dagmc'
DAGMC_INSTALL_DIR=$ROOT_DIR/DAGMC/

# Eigen Install
cd $ROOT_DIR
wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VER}/eigen-${EIGEN_VER}.tar.gz
tar xvf eigen-${EIGEN_VER}.tar.gz
rm eigen-${EIGEN_VER}.tar.gz
cd eigen-${EIGEN_VER}
mkdir build && cd build
cmake ..

# MOAB Install
cd $ROOT_DIR
mkdir MOAB && cd MOAB
git clone -b $MOAB_BRANCH $MOAB_REPO
mkdir build && cd build
cmake ../moab -DENABLE_HDF5=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$MOAB_INSTALL_DIR -DENABLE_BLASLAPACK=OFF -DCMAKE_PREFIX_PATH=$ROOT_DIR/eigen-${EIGEN_VER}
make -j && make -j install
rm -rf $ROOT_DIR/MOAB/moab $HOME/MOAB/build

# DAGMC Install
cd $ROOT_DIR
mkdir DAGMC && cd DAGMC
git clone -b $DAGMC_BRANCH $DAGMC_REPO
mkdir build && cd build
cmake ../dagmc -DBUILD_TALLY=ON -DCMAKE_INSTALL_PREFIX=$DAGMC_INSTALL_DIR -DBUILD_STATIC_LIBS=OFF -DMOAB_DIR=$MOAB_INSTALL_DIR -DEigen3_DIR=$ROOT_DIR/eigen-${EIGEN_VER}/build
make -j install
rm -rf $HOME/DAGMC/dagmc $HOME/DAGMC/build

cd $CURRENT_DIR
