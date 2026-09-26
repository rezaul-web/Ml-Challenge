import pandas as pd
import numpy as np
import time
import yaml
from pathlib import Path
from rapidfuzz import fuzz, distance
import lightgbm as lgb
import re

# ---------------------------------------------------------
# Person B's Module: Feature Engineering & Matching
# Responsibilities: Load candidates and raw data, extract string features, train model, output matches.
# Note: Has its own simple text cleaner to remain completely independent from Person A's module.
# ---------------------------------------------------------

def basic_clean(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()

def load_data_for_features(config):
    s1_path = Path(config['paths']['raw_data_dir']) / "train_source1.tsv"
    s2_path = Path(config['paths']['raw_data_dir']) / "train_source2.tsv"
    s3_path = Path(config['paths']['raw_data_dir']) / "train_source3.tsv"
    
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_pool = pd.concat([
        pd.read_csv(s2_path, sep="\t", dtype=str),
        pd.read_csv(s3_path, sep="\t", dtype=str)
    ])
    
    text_cols = config['preprocessing']['text_columns']
    for df in [df_s1, df_pool]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(basic_clean)
                
    return df_s1.set_index(config['preprocessing']['id_column']), df_pool.set_index(config['preprocessing']['id_column'])

def extract_features(candidates_df, df_s1_idx, df_pool_idx, config):
    print("[Phase 2] Extracting string distance features...")
    start = time.time()
    features = []
    text_cols = config['preprocessing']['text_columns']
    
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_id']
        cand_id = row['candidate_id']
        
        try:
            s1_row = df_s1_idx.loc[s1_id]
            cand_row = df_pool_idx.loc[cand_id]
        except KeyError:
            continue
            
        feat_dict = {
            'source1_id': s1_id,
            'candidate_id': cand_id,
            'blocking_score': row['blocking_score']
        }
        
        for col in text_cols:
            if col in s1_row and col in cand_row:
                t1, t2 = str(s1_row[col]), str(cand_row[col])
                feat_dict[f'{col}_lev'] = distance.Levenshtein.normalized_similarity(t1, t2)
                feat_dict[f'{col}_jaro'] = distance.JaroWinkler.similarity(t1, t2)
                feat_dict[f'{col}_tsort'] = fuzz.token_sort_ratio(t1, t2) / 100.0
                feat_dict[f'{col}_tset'] = fuzz.token_set_ratio(t1, t2) / 100.0
                
        features.append(feat_dict)
        
    print(f"[Phase 2] Feature extraction done in {time.time() - start:.2f}s")
    return pd.DataFrame(features)

def train_and_predict(features_df, config):
    # Dummy implementation for training as actual ground truth processing requires dataset specifics
    print("[Phase 2] Training matching model (Mock)...")
    feature_cols = [c for c in features_df.columns if c not in ['source1_id', 'candidate_id']]
    X = features_df[feature_cols]
    
    # Mocking predictions
    y_pred = np.random.rand(len(X))
    best_threshold = 0.8
    print(f"[Phase 2] Tuned threshold: {best_threshold}")
    return y_pred, best_threshold

def generate_submission(features_df, y_pred, threshold, output_path):
    features_df['pred_prob'] = y_pred
    matches = features_df[features_df['pred_prob'] >= threshold]
    
    submission_records = []
    for s1_id in features_df['source1_id'].unique():
        s1_matches = matches[matches['source1_id'] == s1_id]['candidate_id'].tolist()
        match_str = ",".join(map(str, set(s1_matches))) if s1_matches else ""
        submission_records.append({'source1_id': s1_id, 'matched_ids': match_str})
        
    sub_df = pd.DataFrame(submission_records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"[Phase 2] Final matches saved to {output_path}")

def run_matching_pipeline(config):
    candidates_path = config['paths']['candidate_pairs']
    try:
        candidates_df = pd.read_csv(candidates_path, sep="\t")
    except FileNotFoundError:
        print(f"[Phase 2 Error] {candidates_path} not found. Person A needs to run Candidate Generation first.")
        return False
        
    df_s1_idx, df_pool_idx = load_data_for_features(config)
    features_df = extract_features(candidates_df, df_s1_idx, df_pool_idx, config)
    y_pred, threshold = train_and_predict(features_df, config)
    generate_submission(features_df, y_pred, threshold, config['paths']['matching_results'])
    return True

if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_matching_pipeline(cfg)
