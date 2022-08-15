#!/usr/bin/env bash
poetry export --without-hashes -o requirements.txt
echo "# bigbench" >> requirements.txt
echo "bigbench@https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz" >> requirements.txt
echo "# local package" >> requirements.txt
echo "-e ." >> requirements.txt
