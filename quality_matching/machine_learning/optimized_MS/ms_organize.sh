#!/bin/bash

cd MS_SEG_2008_HF_like/ # MS_SEG_2008 or MS_SEG_2016

pref=1;
for dir in ./*/*/
do
	if [[ "$dir" == */0/* ]]; then
		for file in "$dir"*; do
        	cp "$file" /home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/hf_train/0/$pref-$(basename $file)
        	#echo $pref-"$file"
        done
	fi
    if [[ "$dir" == */1/* ]]; then
        for file in "$dir"*; do
        	cp "$file" /home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/hf_train/1/$pref-$(basename $file)
        	#echo $pref-"$file"
        done
    fi
    #echo $dir
    ((++pref))
done

