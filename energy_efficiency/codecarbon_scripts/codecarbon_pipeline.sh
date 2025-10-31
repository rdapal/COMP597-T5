#!/bin/bash

# this script runs the codecarbon_datacollect.sh script, then the codecarbon_cleanup.sh script and finally any plotting script

# usage: ./codecarbon_pipeline.sh
# NOTE: modify the plotting script to run the one desired

# ------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------ #

# create the needed dirs and log files (for debugging)
# remove the `>> ../codecarbonlogs/*.log 2>&1` if you wish to see the output in the terminal
mkdir -p ../codecarbonlogs
touch ../codecarbonlogs/codecarbon_datacollect.log
touch ../codecarbonlogs/codecarbon_cleanup.log
touch ../codecarbonlogs/plot_losses.log

# run the codecarbon data collection script
echo "Running codecarbon data collection script..."
./codecarbon_datacollect.sh >> ../codecarbonlogs/codecarbon_datacollect.log 2>&1

# run the codecarbon cleanup script
echo "Running codecarbon cleanup script..."
./codecarbon_cleanup.sh >> ../codecarbonlogs/codecarbon_cleanup.log 2>&1

# run the plotting script for losses
echo "Running plotting script for losses..."
./plot_losses.sh >> ../codecarbonlogs/plot_losses.log 2>&1

# run the plotting script for codecarbon data - see code from LAURA
# echo "Running plotting script..."
# ./plots.sh

echo "DONE"

# note: to check the logs as the scripts are running, you can use:
# `tail -n 50 -f ../codecarbonlogs/codecarbon_datacollect.log` where -n is the number of lines you want to see