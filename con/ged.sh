#!/bin/bash

# phuonglh for batch experiments with multiple languages and models
languages=("czech" "english" "german" "italian" "swedish")
outputs=("cs" "en_fce" "de" "it" "sv")

for i in "${!languages[@]}"; do
  # echo $i, ${languages[$i]}, "dat/out/${outputs[$i]}"
  bloop run -p root -m vlp.con.VSC -- -g -l ${languages[$i]} -t ch -m predict -o "dat/out/${outputs[$i]}_ch.tsv" -J-Xmx8g
done
