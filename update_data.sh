#!/usr/bin/env bash
python src/extract_walking_features.py --version V1 --featurize pdkit --filename pdkit_mpower_v1.csv --filtered 
python src/extract_walking_features.py --version V2 --featurize pdkit --filename pdkit_mpower_v2.csv 
python src/extract_walking_features.py --version PASSIVE --featurize pdkit --filename pdkit_mpower_passive.csv 
python src/extract_walking_features.py --version ELEVATE_MS --featurize pdkit --filename pdkit_mpower_elevateMS.csv 
