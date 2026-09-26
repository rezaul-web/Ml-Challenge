import pandas as pd
import numpy as np
import re
import time
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------
# Person A's Module: Candidate Generation
# Responsibilities: Load data, normalize text, perform TF-IDF blocking, save candidate pairs.
# ---------------------------------------------------------

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(filepath, text_cols):
    print(f"[Phase 1] Loading and cleaning {filepath}...")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
            
    df['combined_text'] = df[text_cols[0]]
    if len(text_cols) > 1 and text_cols[1] in df.columns:
         df['combined_text'] = df['combined_text'] + " " + df[text_cols[1]]
         
    return df

def compute_top_k_candidates(tfidf_anchor, tfidf_pool, anchor_ids, pool_ids, top_k, threshold):
    print("[Phase 1] Computing cosine similarities...")
    start_time = time.time()
    similarity_matrix = tfidf_anchor.dot(tfidf_pool.T)
    print(f"[Phase 1] Dot product done in {time.time() - start_time:.2f}s")
    
    candidate_records = []
    for i in range(similarity_matrix.shape[0]):
        row = similarity_matrix.getrow(i)
        indices = row.indices
        data = row.data
        if len(data) == 0: continue
            
        valid_mask = data >= threshold
        indices = indices[valid_mask]
        data = data[valid_mask]
        if len(data) == 0: continue
            
        sorted_idx = np.argsort(-data)[:top_k]
        top_indices = indices[sorted_idx]
        top_scores = data[sorted_idx]
        
        anch_id = anchor_ids[i]
        for pool_idx, score in zip(top_indices, top_scores):
            candidate_records.append({
                'source1_id': anch_id,
                'candidate_id': pool_ids[pool_idx],
                'blocking_score': score
            })
            
    return pd.DataFrame(candidate_records)

def run_candidate_generation(config):
    s1_path = Path(config['paths']['raw_data_dir']) / "train_source1.tsv"
    s2_path = Path(config['paths']['raw_data_dir']) / "train_source2.tsv"
    s3_path = Path(config['paths']['raw_data_dir']) / "train_source3.tsv"
    
    text_cols = config['preprocessing']['text_columns']
    id_col = config['preprocessing']['id_column']
    
    try:
        df_s1 = load_and_preprocess(s1_path, text_cols)
        df_s2 = load_and_preprocess(s2_path, text_cols)
        df_s3 = load_and_preprocess(s3_path, text_cols)
    except FileNotFoundError as e:
        print(f"[Phase 1 Error] {e}. Please ensure data is in {config['paths']['raw_data_dir']}")
        return False
        
    print("[Phase 1] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=tuple(config['blocking']['ngram_range']),
        min_df=2, max_df=0.9
    )
    
    all_text = pd.concat([df_s1['combined_text'], df_s2['combined_text'], df_s3['combined_text']])
    vectorizer.fit(all_text)
    
    tfidf_s1 = vectorizer.transform(df_s1['combined_text'])
    tfidf_s2 = vectorizer.transform(df_s2['combined_text'])
    tfidf_s3 = vectorizer.transform(df_s3['combined_text'])
    
    print("[Phase 1] Matching Source 1 vs Source 2")
    candidates_s1_s2 = compute_top_k_candidates(
        tfidf_s1, tfidf_s2, df_s1[id_col].values, df_s2[id_col].values, 
        config['blocking']['top_k_candidates'], config['blocking']['similarity_threshold']
    )
    
    print("[Phase 1] Matching Source 1 vs Source 3")
    candidates_s1_s3 = compute_top_k_candidates(
        tfidf_s1, tfidf_s3, df_s1[id_col].values, df_s3[id_col].values, 
        config['blocking']['top_k_candidates'], config['blocking']['similarity_threshold']
    )
    
    all_candidates = pd.concat([candidates_s1_s2, candidates_s1_s3], ignore_index=True)
    out_path = config['paths']['candidate_pairs']
    
    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    all_candidates.to_csv(out_path, sep="\t", index=False)
    print(f"[Phase 1] Successfully saved {len(all_candidates)} candidate pairs to {out_path}")
    return True

if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    run_candidate_generation(cfg)
