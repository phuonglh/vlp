#!/bin/bash

# phuonglh for batch experiments with multiple languages and models

languages="czech english german italian swedish"
modelTypes="tk st"

for language in $languages
do
  echo $language
  # tk and st
  for modelType in $modelTypes
  do	  
    bloop run -p root -m vlp.con.VSC -- -g -l $language -m experiment-tk-st -t $modelType
  done
  # ch
  #bloop run -p root -m vlp.con.VSC -- -g -l $language -m experiment-ch -t ch
done

