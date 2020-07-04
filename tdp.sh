#!/bin/bash

spark-submit --driver-memory 256g --class vlp.tdp.Classifier tdp/target/scala-2.11/tdp.jar -l eng -m train -c mlp -f 5 -u 32768 -h 128

