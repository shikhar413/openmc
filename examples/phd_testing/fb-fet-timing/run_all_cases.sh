#!/bin/bash

cd 1d-homog-collision
echo "Running 1d homog with collision-estimator FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 1d-homog -t > cmfd.out
python3 run_openmc_cmfd.py 1d-homog -t
echo "  Running CAPI"
#python3 run_openmc_capi.py 1d-homog -t > capi.out
python3 run_openmc_capi.py 1d-homog -t

cd ../1d-homog-nofet
echo "Running 1d homog with no FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 1d-homog > cmfd.out
python3 run_openmc_cmfd.py 1d-homog
echo "  Running CAPI"
#python3 run_openmc_capi.py 1d-homog > capi.out
python3 run_openmc_capi.py 1d-homog

cd ../1d-homog-nose
echo "Running 1d homog with no FET no SE"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 1d-homog > cmfd.out
python3 run_openmc_cmfd.py 1d-homog
echo "  Running CAPI"
#python3 run_openmc_capi.py 1d-homog > capi.out
python3 run_openmc_capi.py 1d-homog

cd ../1d-homog-fb
echo "Running 1d homog with fission bank FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 1d-homog > cmfd.out
python3 run_openmc_cmfd.py 1d-homog
echo "  Running CAPI"
#python3 run_openmc_capi.py 1d-homog > capi.out
python3 run_openmc_capi.py 1d-homog

cd ../2d-beavrs-collision
echo "Running 2d beavrs with collision-estimator FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 2d-beavrs -t > cmfd.out
python3 run_openmc_cmfd.py 2d-beavrs -t
echo "  Running CAPI"
#python3 run_openmc_capi.py 2d-beavrs -t > capi.out
python3 run_openmc_capi.py 2d-beavrs -t

cd ../2d-beavrs-nofet
echo "Running 2d beavrs with no FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 2d-beavrs > cmfd.out
python3 run_openmc_cmfd.py 2d-beavrs
echo "  Running CAPI"
#python3 run_openmc_capi.py 2d-beavrs > capi.out
python3 run_openmc_capi.py 2d-beavrs

cd ../2d-beavrs-nose
echo "Running 2d beavrs with no FET no SE"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 2d-beavrs > cmfd.out
python3 run_openmc_cmfd.py 2d-beavrs
echo "  Running CAPI"
#python3 run_openmc_capi.py 2d-beavrs > capi.out
python3 run_openmc_capi.py 2d-beavrs

cd ../2d-beavrs-fb
echo "Running 2d beavrs with fission bank FET"
echo "  Running CMFD"
#python3 run_openmc_cmfd.py 2d-beavrs > cmfd.out
python3 run_openmc_cmfd.py 2d-beavrs
echo "  Running CAPI"
#python3 run_openmc_capi.py 2d-beavrs > capi.out
python3 run_openmc_capi.py 2d-beavrs

