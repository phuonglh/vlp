#!/bin/bash

# phuonglh for batch experiments with multiple languages and models
languages=("czech" "english" "german" "italian" "swedish")
outputs=("cs_tk.tsv" "en_fce_tk.tsv" "de_tk.tsv" "it_tk.tsv" "sv_tk.tsv")

for i in "${!languages[@]}"; do
  # echo $i, ${languages[$i]}, "dat/out/${outputs[$i]}"
  bloop run -p root -m vlp.con.VSC -- -g -l ${languages[$i]} -t tk -m predict -o dat/out/${outputs[$i]}
done
