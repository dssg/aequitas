#!/bin/bash
set -e

# download the raw file, unzip and process
#
# raw files are fixed width and come with definition files.
# convert those definition files into in2csv format then
# run in2csv to save in csv format.
#
# input: 8-character zipfile
# outputs: file fixed-width definition
#          data in csv format


# create directories if they don't exist
mkdir -p "data/raw" "data/preprocessed"

ZIPFILE=$1
FILE_NO_EXTENSION="${ZIPFILE%.zip}"
URL="http://www.doc.state.nc.us/offenders"

# download the file
wget -N \
     -P "data/raw/" \
     "$URL"/"$ZIPFILE"

# unzip
unzip -o \
      -d "data/preprocessed/" \
      "data/raw/$ZIPFILE"

# create schema file
echo 'column,start,length' > "data/preprocessed/$FILE_NO_EXTENSION"_schema.csv
in2csv -f fixed \
       -s fixed_width_definitions_format.csv \
       "data/preprocessed/$FILE_NO_EXTENSION".des |
awk '(NR>1)' |
sed -E 's/[ ]{2,}/ /g' |
tr ' ' '_' |
grep -vE "^Name," |
cut -d',' -f2,4-5 >> "data/preprocessed/$FILE_NO_EXTENSION"_schema.csv

# do the conversion 
in2csv -s "data/preprocessed/$FILE_NO_EXTENSION"_schema.csv \
       "data/preprocessed/$FILE_NO_EXTENSION".dat | \
tr -d '?' > "data/preprocessed/$FILE_NO_EXTENSION".csv

