#!/bin/bash

# phuonglh for batch experiments with multiple languages and models

languages = "czech english german italian swedish"

for language in $languages
  # tk and st
  for modelType in "tk st"
    bloop -p root -m vlp.con.VSC -- -g -l $language -m experiment-tk-st -t $modelType
  done
  # ch
  bloop -p root -m vlp.con.VSC -- -g -l $language -m experiment-ch -t ch
done

