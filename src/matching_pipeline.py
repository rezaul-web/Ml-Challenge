import pandas as pd
import numpy as np
import time
import yaml
from pathlib import Path
from rapidfuzz import fuzz, distance
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score

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

def load_ground_truth(config):
    gt_path = config['paths']['train_ground_truth']
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    true_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = str(row['matched_entity_ids'])
        if pd.notna(matched_str) and matched_str.strip() != "":
            matches = [m.strip() for m in matched_str.split(',')]
            for m in matches:
                true_pairs.add((s1_id, m))
    return true_pairs

def f05_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5, zero_division=1.0)

def train_and_predict(features_df, config):
    print("[Phase 2] Training matching model...")
    true_pairs = load_ground_truth(config)
    
    # Create target variable
    features_df['label'] = features_df.apply(
        lambda row: 1 if (row['source1_id'], row['candidate_id']) in true_pairs else 0, 
        axis=1
    )
    
    feature_cols = [c for c in features_df.columns if c not in ['source1_id', 'candidate_id', 'label']]
    X = features_df[feature_cols]
    y = features_df['label']
    
    # Split for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['modeling']['val_size'], random_state=config['modeling']['random_state']
    )
    
    lgb_params = config['modeling']['lgbm_params']
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Tune threshold on validation set
    y_val_probs = model.predict(X_val)
    
    best_thresh = 0.5
    best_f05 = 0.0
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (y_val_probs >= thresh).astype(int)
        f05 = f05_score(y_val, preds)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"[Phase 2] Best validation F0.5 = {best_f05:.4f} at threshold = {best_thresh:.2f}")
    
    # Predict on all data
    y_pred_probs = model.predict(X)
    
    return y_pred_probs, best_thresh

def generate_submission(features_df, y_pred, threshold, df_s1_idx, output_path):
    features_df['pred_prob'] = y_pred
    matches = features_df[features_df['pred_prob'] >= threshold]
    
    submission_records = []
    # Ensure all s1_ids from the original source1 are present (even singletons)
    all_s1_ids = df_s1_idx.index.unique()
    
    for s1_id in all_s1_ids:
        s1_matches = matches[matches['source1_id'] == s1_id]['candidate_id'].tolist()
        match_str = ",".join(map(str, set(s1_matches))) if s1_matches else ""
        submission_records.append({'source1_entity_id': s1_id, 'matched_entity_ids': match_str})
        
    sub_df = pd.DataFrame(submission_records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"[Phase 2] Final matches saved to {output_path}")

def run_matching_pipeline(config):
    # Use the internal candidate pairs file generated by Phase 1
    candidates_path = str(config['paths']['candidate_pairs']).replace('candidate_pairs.tsv', 'internal_candidate_pairs.tsv')
    try:
        candidates_df = pd.read_csv(candidates_path, sep="\t")
    except FileNotFoundError:
        print(f"[Phase 2 Error] {candidates_path} not found. Please run Candidate Generation first.")
        return False
        
    df_s1_idx, df_pool_idx = load_data_for_features(config)
    features_df = extract_features(candidates_df, df_s1_idx, df_pool_idx, config)
    y_pred, threshold = train_and_predict(features_df, config)
    generate_submission(features_df, y_pred, threshold, df_s1_idx, config['paths']['matching_results'])
    return True

if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_matching_pipeline(cfg)
