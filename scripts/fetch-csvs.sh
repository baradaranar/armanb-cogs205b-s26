#!/bin/bash

DATE=$(date +%Y-%m-%d)
TEMPORARY_DIR=$(mktemp -d)
TARGET_DIR="./data/$DATE"

wget -O "$TEMPORARY_DIR/data.zip" "https://github.com/joachimvandekerckhove/cogs205b-s26/raw/9dca64e57fd88213f2422c19a8b10953a8fbfdbe/modules/02-version-control/files/data.zip"
unzip -q "$TEMPORARY_DIR/data.zip" -d "$TEMPORARY_DIR"
mkdir -p "$TARGET_DIR"

CSV_FILES=$(find "$TEMPORARY_DIR" -maxdepth 1 -name "*.csv")
mv $CSV_FILES "$TARGET_DIR/"
rm -rf "$TEMPORARY_DIR"

git add "$TARGET_DIR" 
git add scripts/fetch-csvs.sh
git commit -m "Add CSV data and script on $DATE for Assignment 2"
git push origin main