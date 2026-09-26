import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score
import time

def extract_features(candidates_df, df_source1, df_source2, df_source3, config):
    print("Extracting features...")
    start = time.time()
    
    # Combine Source 2 and 3 for easier lookup
    df_pool = pd.concat([df_source2, df_source3])
    id_col = config['preprocessing']['id_column']
    
    # Set index for fast lookup
    df_source1_idx = df_source1.set_index(id_col)
    df_pool_idx = df_pool.set_index(id_col)
    
    features = []
    
    text_cols = config['preprocessing']['text_columns']
    
    # Iterate over candidates
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_id']
        cand_id = row['candidate_id']
        
        try:
            s1_row = df_source1_idx.loc[s1_id]
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
                s1_text = str(s1_row[col])
                cand_text = str(cand_row[col])
                
                # RapidFuzz features
                feat_dict[f'{col}_levenshtein'] = distance.Levenshtein.normalized_similarity(s1_text, cand_text)
                feat_dict[f'{col}_jaro_winkler'] = distance.JaroWinkler.similarity(s1_text, cand_text)
                feat_dict[f'{col}_token_sort'] = fuzz.token_sort_ratio(s1_text, cand_text) / 100.0
                feat_dict[f'{col}_token_set'] = fuzz.token_set_ratio(s1_text, cand_text) / 100.0
                
        features.append(feat_dict)
        
    print(f"Feature extraction done in {time.time() - start:.2f}s")
    return pd.DataFrame(features)

def calculate_macro_f05(y_true, y_pred_prob, threshold, groups):
    """
    Approximates the macro F0.5 per Source 1 entity.
    groups should be the Source 1 ID.
    Singletons: If true is all 0, and pred is all 0 (below threshold), score=1.0
    If pred has any 1s, score=0.0
    """
    df = pd.DataFrame({
        'group': groups,
        'true': y_true,
        'pred': (y_pred_prob >= threshold).astype(int)
    })
    
    scores = []
    for name, group in df.groupby('group'):
        if group['true'].sum() == 0:
            # True singleton
            if group['pred'].sum() == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            # Not a singleton
            tp = ((group['true'] == 1) & (group['pred'] == 1)).sum()
            fp = ((group['true'] == 0) & (group['pred'] == 1)).sum()
            fn = ((group['true'] == 1) & (group['pred'] == 0)).sum()
            
            if tp == 0:
                scores.append(0.0)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                # F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R)
                f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
                scores.append(f05)
                
    return np.mean(scores)

def find_best_threshold(y_true, y_pred_prob, groups):
    print("Searching for optimal F_0.5 threshold...")
    thresholds = np.linspace(0.5, 0.95, 46) # Test higher thresholds since F0.5 favors precision
    best_t = 0.5
    best_score = -1
    
    for t in thresholds:
        score = calculate_macro_f05(y_true, y_pred_prob, t, groups)
        if score > best_score:
            best_score = score
            best_t = t
            
    print(f"Best Threshold: {best_t:.3f} with F0.5: {best_score:.4f}")
    return best_t

def train_and_predict(features_df, config, ground_truth_df=None):
    # Determine feature columns
    feature_cols = [c for c in features_df.columns if c not in ['source1_id', 'candidate_id', 'label']]
    
    X = features_df[feature_cols]
    
    if ground_truth_df is not None:
        # We need to construct 'label'
        # Assume ground_truth_df has 'source1_id' and 'matched_id'
        # This part requires dataset-specific formatting.
        print("Skipping actual training for now as GT structure is not fully defined.")
        # Mock prediction for structural completeness
        y_pred = np.random.rand(len(X))
        best_threshold = 0.8
    else:
        # Dummy prediction path since model is not trained in this mock
        y_pred = np.random.rand(len(X))
        best_threshold = 0.8
        
    return y_pred, best_threshold

def generate_submission(features_df, y_pred, threshold, output_path):
    features_df['pred_prob'] = y_pred
    matches = features_df[features_df['pred_prob'] >= threshold]
    
    # Format per submission requirements
    # "Output format is strict TSV with exact column names, no duplicate IDs, IDs must exist in test set, every Source 1 entity must appear exactly once."
    
    # We must ensure every Source 1 entity appears exactly once.
    # Group by source1_id and aggregate matches
    submission_records = []
    
    all_source1 = features_df['source1_id'].unique()
    
    for s1_id in all_source1:
        s1_matches = matches[matches['source1_id'] == s1_id]['candidate_id'].tolist()
        
        # Deduplicate IDs
        s1_matches = list(set(s1_matches))
        
        # If empty, it's a singleton. We might need to output empty string or keep it blank depending on specs.
        # Assuming comma-separated for matched IDs as per common formats, wait, problem statement says "ID list columns both contain commas. Read them with explicit tab separator."
        # This implies we write a TSV with source1_id \t matched_ids_comma_separated
        
        match_str = ",".join(map(str, s1_matches)) if s1_matches else ""
        submission_records.append({
            'source1_id': s1_id,
            'matched_ids': match_str
        })
        
    sub_df = pd.DataFrame(submission_records)
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"Submission saved to {output_path}")
