#!/bin/bash

# phuonglh, July 11, 2023
# Decompress OSCAR-2301 files in batches of 10 files.
# command line is a number in range {1..9}
# Usage: ./decompress 1

prefix="$@"
for i in {0..9}
do
    zstd -d "/mnt/nfs/dataset/OSCAR-2301/vi_meta/vi_meta_part_$prefix$i.jsonl.zst" -o "dat/23/$prefix/$prefix$i.jsonl"
done
