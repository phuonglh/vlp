# Pre-processing of the OSCAR corpus

1. Read JSON files from `dat/23`, preprocess and write results to `pre/23`.
2. All documents less than 80 syllables are removed.
3. Multiple lines documents are flatten into lines, removed all lines containing less than 40 characters or more than 2,048 characters.
4. The data frame is de-duplicated by the function "distinct()"  
5. The result directory contains 10 text files.

# Utilities

- Decompress *.zsd files:
    `zstd -d *.jsonl.zst -o *.jsonl`
- Given that there are 9 original data files `dat/23/0*.jsonl`, we run the preprocessing pipeline above as follows:
    `bloop run -p llm -m vlp.llm.OSCAR -- dat/23/0*.jsonl pre/23/0`
- Tokenization:
    `spm_train --input=/home/phuonglh/vlp/llm/pre/23/part-00000-*.txt --model_prefix=oscar --vocab_size=30000 --character_coverage=1.0 --model_type=bpe`
