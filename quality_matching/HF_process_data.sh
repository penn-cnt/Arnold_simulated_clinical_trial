#!/bin/bash
#
# This script takes in a subjects folder containing DICOM images for the Hyperfine research group and converts them into nifti files.
#
# Input:    Calls HF_dicom2nii.sh over all subjects available
#
# Usage:    HF_process_data.sh /path/to/dcm_data/ /path/to/outuput/ 
#
# Example:	./scripts/HF_process_data.sh /Users/tcarnold/Box\ Sync/Stein_Hyperfine/ ./data/NII_version/
#
# Output:   structured Nifti data
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
# 10/7/2019 - created

# make output directory
mkdir ${2}

for folder in $(find ${1}P* -type d)
do
	#echo $folder
	./scripts/HF_dicom2nii.sh ${folder}/ ${2}
done