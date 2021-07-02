#!/bin/bash
#
# This script takes in a subjects folder containing DICOM images for the Hyperfine research group and converts them into nifti files.
#
# Input:    folder for a HF subject from radiology
#
# Usage:    HF_process_data.sh /path/to/832913_Stein_Hyperfine/ 
# Example:	./scripts/HF_process_data.sh /Users/tcarnold/Box\ Sync/832913_Stein_Hyperfine/
#
#
# Output:   structured Nifti data
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
#
# 10/7/2019 - created
# 10/16/2019 - currently working for some subjects, but doesn't have useful names for clinical scans. Want to use sequence info from .json file.

# make output directory
mkdir ${2}

for folder in $(find ${1}P* -type d)
do
	#echo $folder
	./scripts/HF_dicom2nii.sh ${folder}/ ${2}
done