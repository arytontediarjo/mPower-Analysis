#!/usr/bin/env bash
python src/extract_raw_walking_features.py --version MPOWER_V1 --featurize pdkit --filename raw_pdkit_mpower_v1.csv 
python src/extract_raw_walking_features.py --version MPOWER_V2 --featurize pdkit --filename raw_pdkit_mpower_v2.csv 
python src/extract_raw_walking_features.py --version MPOWER_PASSIVE --featurize pdkit --filename raw_pdkit_mpower_passive.csv 
python src/extract_raw_walking_features.py --version MS_ACTIVE --featurize pdkit --filename raw_pdkit_mpower_elevateMS.csv 
python src/clean.py

