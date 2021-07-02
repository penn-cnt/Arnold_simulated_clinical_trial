#!/bin/bash
#
# This script takes in a subjects folder containing DICOM images for the Hyperfine research group and converts them into nifti files.
#
# Input:    folder for a HF subject from radiology
#
# Usage:    HF_dicom2nii.sh data/832913_Stein_Hyperfine/P003_8.13.19/ /path/to/output/
#
# Steps:    1. get filename information and make output directory
#			2. unzip main folder and subfolders
#			Process clinical data
#			3. convert files with no extension to dicom files.
#			4. convert dicom to nifti files
#			5. move to output directory
#			Process Hyperfine data
#			6. convert dicom to nifti files
#			7. move to output directory
#			8. remove DCM images/folders
#
# Output:   structured Nifti data
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
#
# 10/7/2019 - created
# 10/16/2019 - currently working for some subjects, but doesn't have useful names for clinical scans. Want to use sequence info from .json file.

#### step 1 ####

# get folder name
filename=$(basename -- "$1")
MAIN_NAME="${filename%/}"

# setup output structure
OUT_DIR=./data/$MAIN_NAME/
DCM_DIR=./data/${MAIN_NAME}_dcm/
mkdir $OUT_DIR
mkdir $DCM_DIR

echo $MAIN_NAME
echo $1
echo ${OUT_DIR}

#### step 2 ####

# unzip main folder
unzip -q -d $DCM_DIR $1 

# loop through and unzip subfolder
for folder in $1*.zip
do
	unzip -q -d $DCM_DIR $folder
done

# get names of present folder (will delete later)
dcm_folders=$(find ${DCM_DIR}/*/ -type d)

#### CLINICAL IMAGING  ####

#### step 3 ####

# convert and extensionless files into DICOM extensions
for file in $(find $DCM_DIR/*/DICOM/* -type f)
do
	mv ${file} ${file}.dcm
done

#### step 4 ####

# convert DICOM to Nifti
for folder in $(find $DCM_DIR/*/DICOM/* -type d)
do
	var=$(ls ${folder}/*.dcm) # get all dicom files in folder
	N=1 # index of file to read in
	file=$(echo $var | cut -d " " -f $N) # select the first file name
	dcm2niix -z y ${file} # make dicom file
done

#### step 5 ####

# grab Nifti files and put in new folder
N=4
for file in $(find ${DCM_DIR}*/DICOM/ -type f -name "*.nii.gz")
do
	date_folder=$(echo $file | cut -d "/" -f $N)
	mkdir ${OUT_DIR}/${date_folder}/
	mv ${file} ${OUT_DIR}/${date_folder}/
done

# grab JSON files and put in new folder
for file in $(find ${DCM_DIR}*/DICOM/ -type f -name "*.json")
do
	date_folder=$(echo $file | cut -d "/" -f $N)
	mkdir ${OUT_DIR}/${date_folder}/
	mv ${file} ${OUT_DIR}/${date_folder}/
done

# rename all files using descriptions from CONTENT.XML
for folder_path in ${DCM_DIR}*
do
	folder_name=$(basename -- "$folder_path")
	echo ${folder_path} 
	echo $folder_name
	python3 ./scripts/HF_rename_clinical.py ${OUT_DIR}${folder_name} ${DCM_DIR}${folder_name} 
done

#### HYPERFINE ####

#### step 6 ####

# convert DICOM to Nifti
file=$(find ${DCM_DIR}*/DICOMDIR/* -type f)
dcm2niix -z y ${file} # Just dcm2niix the last file, should convert all

#### step 7 ####

# grab Nifti files and put in new folder
for file in $(find ${DCM_DIR}*/DICOMDIR/ -type f -name "*.nii.gz")
do
	date_folder=$(echo $file | cut -d "/" -f $N)
	mkdir ${OUT_DIR}/${date_folder}/
	mv ${file} ${OUT_DIR}/${date_folder}/
done

# grab Nifti files and put in new folder
for file in $(find ${DCM_DIR}*/DICOMDIR/ -type f -name "*.json")
do
	date_folder=$(echo $file | cut -d "/" -f $N)
	mkdir ${OUT_DIR}/${date_folder}/
	mv ${file} ${OUT_DIR}/${date_folder}/
done

# remove DICOMDIR_ prefix
for file in $(find ${OUT_DIR} -type f)
do
	suffix="${file##*DICOMDIR_}"
	mv $file ${foldername}/${filename}/$suffix
done

#### step 8 ####

# remove unzipped folders containing dcm
rm -r ${DCM_DIR}
