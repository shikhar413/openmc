#!/bin/bash
#Set FET filter type to analog instead of collision in src/tallies/tally.cpp

cd 1d-homog-analog
echo "Running 1d homog with collision estimator FET"
echo "  Running CMFD"
python3 run_openmc_cmfd.py 1d-homog -t > cmfd.out
echo "  Running CAPI"
python3 run_openmc_capi.py 1d-homog -t > capi.out

cd ../2d-beavrs-analog
echo "Running 2d beavrs with collision estimator FET"
echo "  Running CMFD"
python3 run_openmc_cmfd.py 2d-beavrs -t > cmfd.out
echo "  Running CAPI"
python3 run_openmc_capi.py 2d-beavrs -t > capi.out
