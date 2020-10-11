# Named Entity Recognition for LAD

## Data Preparation for LAD from HUS Tagging Group

- The final resulting file is in `dat/ner/lad-b*.tsv`. This file is in the 2-col format.
- First, run the `rename` method of `Utils.scala` to rename folders/files to tsv files, which are stored in `dat/ner/stm`
- Then, run the `convert` method of `Utils.scala` to convert 4-col format files in `dat/ner/stm` into 2-col format files, which are stored in `dat/ner/lad`. This conversion use BIO format for entities.
- Finally, run the `CorpusReader.scala` file to combine and filter all labeled sentences in `dat/ner/lad/b*` into a 2-col format file `dat/ner/lad-b*.tsv`.

## Training a Bidirectional CMM

- Train the forward model:
    `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j` (note the last '/' at the end of model path)
- Train the backward model:
    `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j -r`

## Evaluating the Model

  Use the 'eval' mode: `bloop run -m vlp.ner.Tagger ner -- -m eval -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j`

  This will create the file `dat/ner/lad.tsv.out`. Then, one can use the `conlleval` script to evaluate the tagging result:

  `conlleval < lad.tsv.out` 

```
processed 202819 tokens with 8286 phrases; found: 8552 phrases; correct: 7527.
accuracy:  96.21%; precision:  88.01%; recall:  90.84%; FB1:  89.40
             DATE: precision:  91.64%; recall:  93.18%; FB1:  92.40  2190
              DOC: precision:  88.59%; recall:  86.81%; FB1:  87.69  1323
              LOC: precision:  72.29%; recall:  82.38%; FB1:  77.01  841
              ORG: precision:  88.69%; recall:  92.38%; FB1:  90.50  3882
              PER: precision:  93.99%; recall:  93.69%; FB1:  93.84  316
```

## Data Preparation for LAD from BPO Group

- The resulting file is in `dat/ner/bpo/man.tsv`. This file is the result of converting files in `dat/ner/bpo/man/` directory.
- To get the resulting file, run the `convertBPO` method of `Utils.scala` to convert XML format files in `dat/ner/bpo/man` into 2-col format.

## Training a Bidirectional CMM for BPO (Convert from XML format like the VLSP 2018 corpus)

  `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/bpo/man.tsv -p /opt/models/ner/bpo/ -j`
  `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/bpo/man.tsv -p /opt/models/ner/bpo/ -j -r`

## Eval on BPO

  `bloop run -m vlp.ner.Tagger ner -- -m eval -d dat/ner/bpo/man.tsv -p /opt/models/ner/bpo/ -j`

```
./conlleval < bpo/man.tsv.out 
processed 25818 tokens with 2120 phrases; found: 1855 phrases; correct: 1752.
accuracy:  96.97%; precision:  94.45%; recall:  82.64%; FB1:  88.15
              DOC: precision:  93.27%; recall:  24.49%; FB1:  38.80  104
              LOC: precision:  81.84%; recall:  87.68%; FB1:  84.66  435
              ORG: precision:  98.97%; recall:  98.88%; FB1:  98.92  1160
              PER: precision:  96.79%; recall:  96.18%; FB1:  96.49  156
```  

If enable "regexp" features:

```
processed 25818 tokens with 2120 phrases; found: 1842 phrases; correct: 1745.
accuracy:  97.10%; precision:  94.73%; recall:  82.31%; FB1:  88.09
              DOC: precision:  92.86%; recall:  22.98%; FB1:  36.84  98
              LOC: precision:  83.41%; recall:  87.93%; FB1:  85.61  428
              ORG: precision:  98.79%; recall:  98.62%; FB1:  98.71  1159
              PER: precision:  96.82%; recall:  96.82%; FB1:  96.82  157
```              

If do not tokenize "DOC" entity type:

```
phuonglh@PhuongLHMacBookPro ner % ./conlleval < bpo/man.tsv.out
processed 26712 tokens with 2120 phrases; found: 2147 phrases; correct: 2050.
accuracy:  98.54%; precision:  95.48%; recall:  96.70%; FB1:  96.09
              DOC: precision:  98.98%; recall:  98.48%; FB1:  98.73  394
              LOC: precision:  82.30%; recall:  88.18%; FB1:  85.14  435
              ORG: precision:  99.05%; recall:  99.14%; FB1:  99.10  1162
              PER: precision:  96.79%; recall:  96.18%; FB1:  96.49  156
```              

## Eval on BPO+LAD-b1, a total of 2,972 annotated sentences (see the file dat/ner/man/man.tsv)

processed 99428 tokens with 8926 phrases; found: 9042 phrases; correct: 8528.
accuracy:  98.32%; precision:  94.32%; recall:  95.54%; FB1:  94.92
             DATE: precision:  95.07%; recall:  98.49%; FB1:  96.75  1786
              DOC: precision:  99.53%; recall:  99.47%; FB1:  99.50  1505
              LOC: precision:  80.35%; recall:  85.92%; FB1:  83.04  1033
              ORG: precision:  95.46%; recall:  95.57%; FB1:  95.52  4384
              PER: precision:  94.91%; recall:  90.31%; FB1:  92.55  334

## Spark JobServer Installation

## Install Service

  `curl -X POST localhost:8090/binaries/ner -H "Content-Type: application/java-archive" --data-binary @ner.jar`
  `curl -X POST localhost:8090/binaries/sjs -H "Content-Type: application/java-archive" --data-binary @sjs.jar`

  `curl -d "dependent-jar-uris=[ner]" "localhost:8090/contexts/ner?num-cpu-cores=8&memory-per-node=1024m"`

## Submit Request
  
- General form:  
  `curl -d "text=\"...\"" -H "Content-Type: text/plain; charset=UTF-8"  "localhost:8090/jobs?appName=ner&classPath=vlp.ner.TaggerJob&context=ner&sync=true"`
- Example 1: single sentence
  `curl -d "text=\"Đồng thời kiểm tra, xác minh nội dung đơn của ông Nguyễn Văn Út về việc Trường Tiểu học Cầu Sơn lấn chiếm phần đất có nguồn gốc do Đình Cầu Sơn sử dụng trước đây (hiện nay Trường Tiểu học Cầu Sơn đã hoán đổi cho Công ty phát triển nhà quận Bình Thạnh) báo cáo kết quả cho Ủy ban nhân dân Thành phố.\"" -H "Content-Type: text/plain; charset=UTF-8"  "localhost:8090/jobs?appName=sjs&classPath=vlp.ner.TaggerJob&context=ner&sync=true"`
- Example 2: multiple sentences
  `curl -d "text=\"Tiêu huỷ toàn bộ số lượng thuốc lá tịch thu nêu trên. Xử lý theo Quyết định số 2371/QĐ-TTg ngày 26 tháng 12 năm 2014 của Thủ tướng Chính phủ về việc thực hiện tiêu huỷ thuốc lá nhập lậu bị tịch thu; Điều 2. Ông Huỳnh Thanh Dũng phải nghiêm chỉnh chấp hành quyết định xử phạt vi phạm hành chính trong thời hạn là 10 (mười) ngày, kể từ ngày được giao Quyết định xử phạt. Quá thời hạn này, ông Huỳnh Thanh Dũng cố tình không chấp hành Quyết định xử phạt thì bị cưỡng chế thi hành theo quy định và cứ mỗi ngày chậm nộp phạt thì phải nộp thêm 0,05% trên tổng số tiền phạt chưa nộp. Số tiền phạt quy định tại Điều 1 phải nộp tại Chi cục Quản lý thị trường thành phố, số 02 đường Phạm Ngũ Lão, phường Phạm Ngũ Lão, quận 1, thành phố Hồ Chí Minh.\"" -H "Content-Type: text/plain; charset=UTF-8"  "localhost:8090/jobs?appName=sjs&classPath=vlp.ner.TaggerJob&context=ner&sync=true"`
