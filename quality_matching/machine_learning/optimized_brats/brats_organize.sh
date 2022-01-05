#!/bin/bash

cd BraTS_2019_HF_like/HGG/

pref=1;
for dir in ./*/*/
do
	if [[ "$dir" == */0/* ]]; then
		for file in "$dir"*; do
        	cp "$file" /home/steve/PycharmProjects/hyperfine/optimized_brats/HGG/hf_train/0/$pref-$(basename $file)
        	#echo $pref-"$file"
        done
	fi
    if [[ "$dir" == */1/* ]]; then
        for file in "$dir"*; do
        	cp "$file" /home/steve/PycharmProjects/hyperfine/optimized_brats/HGG/hf_train/1/$pref-$(basename $file)
        	#echo $pref-"$file"
        done
    fi
    #echo $dir
    ((++pref))
done

