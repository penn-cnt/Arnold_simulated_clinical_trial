# This script takes in an XML output from SECTRA and extracts interpretable descriptions of the imaging content. 
#
# Input:    CONTENT.XML
#
# Usage:    python3 HF_xml_parse.py SECTRA_XML_file
#
# Example:  python3 HF_xml_parse.py CONTENT.XML
#
# Output:   dictornary {study ID: description}
#
# Thomas Campbell Arnold 
# tcarnold@seas.upenn.edu
# 2/19/2021 - created

import xml.etree.ElementTree as ET
import sys

xml_filename = sys.argv[1]
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
