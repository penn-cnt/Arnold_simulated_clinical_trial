#!/bin/bash
#
# This script takes in a subjects .zip file containing DICOM images for the Hyperfine research group and converts them into nifti files.
#
# Input:    zip folder of subject data provided by neuroradiology
#
# Usage:    HFzip2nii.sh data/DCM_version/subject_folder.zip
#
# Usage:    HFzip2nii.sh data/DCM_version/P012.zip
#
# Output:   structured Nifti data
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
#
# 10/7/2019 - created
# 02/10/2021 - modified from HF_dicom2nii.sh

# get folder and file info
filepath=$1
foldername=$(dirname "$filepath")
filename=$(basename -- "$filepath")
extension="${filename##*.}"
filename="${filename%.*}"

# print out folder and file names for troubleshooting
echo $filename
echo $extension
echo $foldername
echo ${foldername}/${filename}

# unzip main folder
mkdir ${foldername}/${filename}
unzip -q -d ${foldername}/${filename}/ ${filepath}

# convert to NII, move to top directory, remove dcm files
dcm2niix -z y ${foldername}/${filename}/p*/DICOMDIR/*
mv ${foldername}/${filename}/p*/DICOMDIR/*.nii.gz ${foldername}/${filename}/
rm ${foldername}/${filename}/p*/DICOMDIR/*
rmdir ${foldername}/${filename}/p*/DICOMDIR
rmdir ${foldername}/${filename}/p*

# remove DICOMDIR_ prefix
for file in $(find ${foldername}/${filename} -type f)
do
	suffix="${file##*DICOMDIR_}"
	mv $file ${foldername}/${filename}/$suffix
done
