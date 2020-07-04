#!/bin/bash

spark-submit --driver-memory 28g --class vlp.vec.W2V vec/target/scala-2.11/vec.jar -m train -i /opt/data/fin/ -o dat/vec/fin/vie.fin.50d.txt -p dat/vec/fin

