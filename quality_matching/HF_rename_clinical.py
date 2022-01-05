# This script takes in an XML output from SECTRA and extracts interpretable descriptions of the imaging content. 
#
# Input:    CONTENT.XML
#
# Usage:	python3 ./scripts/HF_rename_clinical.py subject_folder subject_dcm_folder
#
# Example: 	python3 ./scripts/HF_rename_clinical.py ./data/P033/P033_clinical ./data/P033_dcm/P033_clinical
#
# Output:   dictornary {study ID: description}
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
# 2/19/2021 - created

import os
import sys
import xml.etree.ElementTree as ET

def HF_xml_parse(xml_filename):

	# xml_filename = sys.argv[1]
	tree = ET.parse(xml_filename)
	root = tree.getroot()

	# build dict with study ID and description of image
	study_des = {}
	for studies in root.iter('study'):
	    for study in studies:
	        # print(study.attrib) # print study 
	        for des in study.iter('description'):
	        	# print(des.tag, des.text)
	        	study_des[study.attrib.get('id')] = des.text
	# print(study_des)

	# remove none values, remove all spaces and replace with _
	study_des = {k:v for k,v in study_des.items() if v is not None}
	study_des = {k:v.replace(' ','_') for k,v in study_des.items()}

	# print out study and descriptions
	for k,v in study_des.items():
		print(k, v,)

	return study_des

##### MAIN #####

# get all filenames of all nii/json files
files = []
for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    files.extend(filenames)
print(files)

# get list of all study IDs
study_ID = []
for file in files:
	study_ID.append(file[:8])
print(study_ID)

# get dictionary of all study IDs and descriptions
xml_filename = os.path.join(sys.argv[2], 'SECTRA', 'CONTENT.XML')
print(xml_filename)
study_des = HF_xml_parse(xml_filename)
print(study_des)

# loop through eacg file and print out description
for file in files:
	#print(file, study_des[file[:8]])
	src = os.path.join(sys.argv[1], file)
	print(src)

	# get file extension
	if src[-5:] == '.json':
		ext = '.json'
	else:
		ext = '.nii.gz'

	# get destination
	dest = os.path.join(sys.argv[1], study_des[file[:8]] + ext)
	print(dest)

	# rename the files
	os.rename(src, dest)
