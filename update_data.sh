#!/usr/bin/env bash
python src/extract_raw_walking_features.py --version V1 --featurize pdkit --filename pdkit_mpower_v1.csv --filtered --update
python src/extract_raw_walking_features.py --version V2 --featurize pdkit --filename pdkit_mpower_v2.csv --update
python src/extract_raw_walking_features.py --version PASSIVE --featurize pdkit --filename pdkit_mpower_passive.csv --update
python src/extract_raw_walking_features.py --version EMS --featurize pdkit --filename pdkit_mpower_elevateMS.csv --update
python src/clean.py
python src/create_model_data.py
