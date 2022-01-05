#!/bin/bash
#
# Input:    3T clinical image
#           Hyperfine image
#
# Usage:    quality_match.sh 3T.nii Hyperfine.nii
#
# Steps:    1. register 3T image to Hyperfine image
#           2. transfrom 3T to Hyperfine space
#           3. extract brain of 3T and Hyperfine
#           4. run quality matching algorithm
#
# Output:   Hyperfine-like image based on 3T transformation and relevant transformation parameters
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
#
# 10/2/2019 - created

# get filenames
filename=$(basename -- "$1")
T3ext="${filename##*.}"
T3name="${filename%.*}"
filename=$(basename -- "$2")
HFext="${filename##*.}"
HFname="${filename%.*}"
T3full=${T3name}.${T3ext}
HFfull=${HFname}.${HFext}

# set output directory
OUT_DIR=./analysis/${T3name}/
ATLAS_DIR=./tools/ADNI_normal_atlas/
mkdir ${OUT_DIR}

# assign filenames to meaningful variables
T3=${1}
HF=${2}

# center images on common origin
c3d $T3 -origin-voxel 50% -o ${OUT_DIR}${T3full}
c3d $HF -origin-voxel 50% -o ${OUT_DIR}${HFfull}

# brain extraction
antsBrainExtraction.sh -d 3 -a ${OUT_DIR}${T3full} -e ${ATLAS_DIR}T_template0.nii.gz -m ${ATLAS_DIR}T_template0_BrainCerebellumProbabilityMask.nii.gz -f ${ATLAS_DIR}T_template0_BrainCerebellumExractionMask.nii.gz -o ${OUT_DIR}T3_
antsBrainExtraction.sh -d 3 -a ${OUT_DIR}${HFfull} -e ${ATLAS_DIR}T_template0.nii.gz -m ${ATLAS_DIR}T_template0_BrainCerebellumProbabilityMask.nii.gz -f ${ATLAS_DIR}T_template0_BrainCerebellumExractionMask.nii.gz -o ${OUT_DIR}HF_

# rigid & affine registration
antsRegistration \
--dimensionality 3 \
--float 0 \
--output ${OUT_DIR}T32HF_ \
--interpolation Linear \
--use-histogram-matching 0 \
--initial-moving-transform [${OUT_DIR}HF_BrainExtractionBrain.nii.gz,${OUT_DIR}T3_BrainExtractionBrain.nii.gz,1] \
--transform Rigid[0.1] \
--metric MI[${OUT_DIR}HF_BrainExtractionBrain.nii.gz,${OUT_DIR}T3_BrainExtractionBrain.nii.gz,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox \
--transform Affine[0.1] \
--metric MI[${OUT_DIR}HF_BrainExtractionBrain.nii.gz,${OUT_DIR}T3_BrainExtractionBrain.nii.gz,1,32,Regular,0.25] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox 

# transform 3T to Hyperfine space
antsApplyTransforms \
-d 3 \
-i ${OUT_DIR}T3_BrainExtractionBrain.nii.gz \
-o ${OUT_DIR}T32HF_T3_BrainExtractionBrain.nii.gz \
-t ${OUT_DIR}T32HF_0GenericAffine.mat \
-r ${OUT_DIR}${HFfull} \
-n Linear

# run quality matching in matlab
matlab -nodesktop -nosplash -r "addpath(genpath('scripts')); quality_match_ex('"${OUT_DIR}"HF_BrainExtractionBrain.nii.gz','"${OUT_DIR}"T32HF_T3_BrainExtractionBrain.nii.gz','"${OUT_DIR}"'); exit;"





