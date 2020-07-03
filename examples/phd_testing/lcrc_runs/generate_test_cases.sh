#!/bin/bash

#set -ex

cluster=lcrc
run_scripts=true

if [ "$#" -ne 3 ]; then
  echo "usage: prog [problem] [seed_start] [seed_end]"
else
  prob=$1
  begin=$2
  end=$3

  cd $prob
  for ((seed=$begin; seed<=$end; seed++))
  do
    mkdir -p seed$seed
    cp base/*.xml seed$seed
    cp base/*.py seed$seed
    cp base/batch_script_"$cluster".slurm seed$seed
    cd seed$seed
    sed -i s/{seed}/$seed/g batch_script_$cluster.slurm
    sed -i s/{seed}/$seed/g settings.xml
    if $run_scripts ; then
      echo "Running job for" $prob "seed" $seed
      sbatch batch_script_$cluster.slurm
    fi
    cd ..
  done
fi
