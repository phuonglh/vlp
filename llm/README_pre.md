# Pre-processing of the OSCAR corpus

    1. Read JSON files (for 22/23/c4 versions), TEXT files for 21 version.
    2. All documents less than 80 syllables are removed.
    3. Multiple lines documents are flatten into lines, removed all lines containing less than 40 characters or more than 2,048 characters.
    4. The data frame is de-duplicated by the function "distinct()"  
    5. Each output directory contains 10 gzip JSON files.

    Documents => DocumentFilter => LineFlatten => LineFilter => LineDeduplication => Save

## Exact Document Deduplication

Choose an OSCAR version in the set [21, 23, c4, 2x]. In the A100-NLP server, for a run of trunk 900 of C4 corpus:

    `bloop run -p llm -m vlp.llm.OSCAR -- -l d -v c4 -i /mnt/nfs/dataset/OSCAR/c4/900 -o /home/phuonglh9/c4/900 -X 64 -Y 16 -D 32g -E 20g`

Here, we have 4 executors, each requires 20g memory to run. This command performs deduplication at document level. 

| Trunk  | #(documents) |
|--------|--------------|
| c4/900 | 7,498,976    |
| c4/800 | 7,498,541    | 
| c4/700 | 7,498,253    | 
| c4/600 | 7,498,617    |
| c4/500 | 7,498,796    |
| c4/400 | 7,498,629    |
| c4/300 | 7,499,398    |
| c4/200 | 7,498,559    |
| c4/100 | 7,497,753    |
| c4/000 | 7,498,424    |
| c4/x00 | 1,799,376    |

In the A100-CV server, to perform deduplication of OSCAR 21 at the document level:    

    - `bloop run -p llm -m vlp.llm.OSCAR -- -l d -v 21 -i /data/dataset/OSCAR/21/text -o /data/nlp.phuonglh/OSCAR/21 -X 64 -Y 16 -D 32g -E 20g`
    - There are 13,470,695 unique documents. About 13GB of gzip JSON.

For version 22:

    -`bloop run -p llm -m vlp.llm.OSCAR -- -l d -v 23 -i /data/dataset/OSCAR/22/ -o /data/nlp.phuonglh/OSCAR/22 -X 64 -Y 16 -D 32g -E 20g`
    - There are 7,687,667 unique documents. About 16GB of gzip JSON.

For version 23:

    - `bloop run -p llm -m vlp.llm.OSCAR -- -l d -v 23 -i /data/dataset/OSCAR/23/ -o /data/nlp.phuonglh/OSCAR/23 -X 64 -Y 16 -D 32g -E 20g`
    - There are 13,600,222 unique documents. About 29GB of gzip JSON.

## Exact Paragraph Deduplication

Output directory on A100-CV: `~/vlp/llm/par/(21|22|23)`.

To perform deduplication of OSCAR 21 at the paragraph level on the A100-CV server:

    - `bloop run -p llm -m vlp.llm.OSCAR -- -l p -i /data/nlp.phuonglh/OSCAR/21/ -o par/21  -X 64 -Y 16 -D 32g -E 20g`
    - There are 114,530,056 input paragraphs in 13,470,695 documents. 
    - There are 114,481,785 output paragraphs in 13,467,638 documents.

For version 22:

    - `bloop run -p llm -m vlp.llm.OSCAR -- -l p -i /data/nlp.phuonglh/OSCAR/22/ -o par/22  -X 64 -Y 16 -D 32g -E 20g`
    - There are 232,721,195 input paragraphs in 6,651,437 documents.
    - There are 117,772,609 output paragraphs in 6,402,758 documents.

For version 23: 

    - `bloop run -p llm -m vlp.llm.OSCAR -- -l p -i /data/nlp.phuonglh/OSCAR/23/ -o par/23  -X 64 -Y 16 -D 32g -E 20g`
    - There are 439,273,347 input paragraphs in 11,979,643 documents.
    - There are 215,785,382 output paragraphs in 11,579,078 documents.


| Corpus | #(inpParagraphs) | #(outParagraphs) | #(outDocuments) | Size (GB gzip json) |
|--------|------------------|------------------|-----------------|---------------------| 
| 21     | 114,530,056      | 114,481,785      | 13,467,638      | 13.0                |
| 22     | 232,721,195      | 117,772,609      | 06,402,758      | 09.8                |           
| 23*    | 439,273,347      | 215,785,382      | 11,579,078      | 18.0                |


For version c4, to perform deduplication of OSCAR 23 chunk x00 at the paragraph level, on the A100-NLP server:

`bloop run -p llm -m vlp.llm.OSCAR -- -l p -i /home/phuonglh9/c4/x00 -o /home/phuonglh9/c4-par/x00 -X 64 -Y 16 -D 32g -E 20g`

Repeat the command for all other trunks (100|...|900).

| Trunk  | #(inpParagraphs) | #(outParagraphs) | #(outDocuments) |
|--------|------------------|------------------|-----------------| 
| c4/000 | 140,295,673      | 140,294,056      | 7,498,424       |
| c4/100 | 140,212,973      | 140,211,668      | 7,497,753       |
| c4/200 | 140,522,569      | 140,521,369      | 7,498,559       |
| c4/300 | 140,382,280      | 140,381,411      | 7,499,397       |
| c4/400 | 140,232,237      | 140,230,182      | 7,498,629       |
| c4/500 | 140,396,826      | 140,394,975      | 7,498,796       |
| c4/600 | 140,320,572      | 140,319,393      | 7,498,617       |
| c4/700 | 140,475,618      | 140,474,775      | 7,498,253       |
| c4/800 | 140,370,570      | 140,367,872      | 7,498,541       |
| c4/900 | 140,546,151      | 140,544,799      | 7,498,976       |
| c4/x00 | 033,700,269      | 033,700,191      | 1,799,376       |

The `~/c4` directory contains document-level deduplicated corpus. The `~/c4-par` contains paragraph-level deduplicated corpus.
There are about 76.8M unique documents, 94 GB of gzip JSON. 

Perform deduplication of all the trunks using 2 executors, each has 45g RAM:

`bloop run -p llm -m vlp.llm.OSCAR -- -l p -i /home/phuonglh9/c4-par -o /home/phuonglh9/c4-par-all -X 64 -Y 32 -D 32g -E 45g`

There are 1,437,440,680 paragraphs in 76,785,321 documents. 

To combine all 21, 22, 23 p-level documents into one big chunk on A100-NLP (192GB of RAM)

`bloop run -p llm -m vlp.llm.OSCAR -- -l d -v 2x -i par/ -o par-2x -X 64 -Y 32 -D 64g -E 45g`

To dedup all the C4 results (second times) from the Spark Cluster using 6 executors

spark-3.4.0-bin-hadoop3/bin/spark-submit --conf spark.local.dir=/mnt/a100-llm/tmp  --class vlp.llm.OSCAR --master spark://103.176.147.153:7077 vlp/llm/target/scala-2.12/llm.jar -l p -i /mnt/a100-llm/llm-corpora/c4-par/ -o /mnt/a100-llm/llm-corpora/c4-par-all -X 60 -Y 240 -D 320g -E 480g -M spark://103.176.147.153:7077 -t /mnt/a100-llm/tmp 

    
# Tokenization

    `spm_train --input=/home/phuonglh/vlp/llm/pre/23/part-00000-*.txt --model_prefix=oscar --vocab_size=30000 --character_coverage=1.0 --model_type=bpe`

# Excrawl Stats:

Forum:
    Number of input documents = 158031
    Unique d-level documents = 121287
    Number of input paragraphs = 1264352
    Number of output paragraphs = 857545
    Unique p-level documents  = 111996
    Good p-level documents = 111662

