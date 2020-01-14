#!/bin/bash

#set -ex
run_scripts=true
threads=(1 2 4 8 16)

prob=assembly
for t in ${threads[@]}; do
    dir=2d-beavrs-$prob-"$t"threads
    mkdir -p $dir
    cp base/* $dir/
    sed -i s/{n_threads}/$t/g $dir/batch_script.slurm
    sed -i s/{prob}/$prob/g $dir/batch_script.slurm
    if $run_scripts ; then
      cd $dir
      echo "Running job for" $prob "thread" $t
      sbatch batch_script.slurm
      cd ..
    fi
done

prob=qassembly
for t in ${threads[@]}; do
    dir=2d-beavrs-$prob-"$t"threads
    mkdir -p $dir
    cp base/* $dir/
    sed -i s/{n_threads}/$t/g $dir/batch_script.slurm
    sed -i s/{prob}/$prob/g $dir/batch_script.slurm
    if $run_scripts ; then
      cd $dir
      echo "Running job for" $prob "thread" $t
      sbatch batch_script.slurm
      cd ..
    fi
done

