import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import time

def compute_top_k_candidates(tfidf_anchor, tfidf_pool, anchor_ids, pool_ids, top_k, threshold):
    """
    Computes cosine similarity between anchor and pool using sparse matrix dot product.
    Returns a dataframe of candidate pairs.
    """
    print("Computing cosine similarities...")
    start_time = time.time()
    
    # Dot product for cosine similarity (assuming normalized vectors, which TfidfVectorizer does by default)
    similarity_matrix = tfidf_anchor.dot(tfidf_pool.T)
    
    print(f"Dot product done in {time.time() - start_time:.2f}s")
    
    candidate_records = []
    
    # Process row by row to find top K
    # For large datasets, this can be parallelized or optimized further
    for i in range(similarity_matrix.shape[0]):
        row = similarity_matrix.getrow(i)
        
        # Get indices and data of non-zero elements
        indices = row.indices
        data = row.data
        
        if len(data) == 0:
            continue
            
        # Filter by threshold
        valid_mask = data >= threshold
        indices = indices[valid_mask]
        data = data[valid_mask]
        
        if len(data) == 0:
            continue
            
        # Sort and take top K
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

def generate_candidates(df_source1, df_source2, df_source3, config):
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=tuple(config['blocking']['ngram_range']),
        min_df=2, # slightly filter extreme noise
        max_df=0.9 # remove very common ngrams
    )
    
    # Fit on all data to ensure common vocabulary
    all_text = pd.concat([
        df_source1['combined_text'], 
        df_source2['combined_text'], 
        df_source3['combined_text']
    ])
    vectorizer.fit(all_text)
    
    print("Transforming Source 1...")
    tfidf_s1 = vectorizer.transform(df_source1['combined_text'])
    
    print("Transforming Source 2...")
    tfidf_s2 = vectorizer.transform(df_source2['combined_text'])
    
    print("Transforming Source 3...")
    tfidf_s3 = vectorizer.transform(df_source3['combined_text'])
    
    id_col = config['preprocessing']['id_column']
    
    # S1 vs S2
    print("Matching Source 1 vs Source 2")
    candidates_s1_s2 = compute_top_k_candidates(
        tfidf_s1, tfidf_s2, 
        df_source1[id_col].values, df_source2[id_col].values, 
        config['blocking']['top_k_candidates'], 
        config['blocking']['similarity_threshold']
    )
    candidates_s1_s2['source'] = 'source2'
    
    # S1 vs S3
    print("Matching Source 1 vs Source 3")
    candidates_s1_s3 = compute_top_k_candidates(
        tfidf_s1, tfidf_s3, 
        df_source1[id_col].values, df_source3[id_col].values, 
        config['blocking']['top_k_candidates'], 
        config['blocking']['similarity_threshold']
    )
    candidates_s1_s3['source'] = 'source3'
    
    # Combine
    all_candidates = pd.concat([candidates_s1_s2, candidates_s1_s3], ignore_index=True)
    return all_candidates

def evaluate_blocking(candidates_df, ground_truth_df):
    """
    Evaluates blocking recall. This is a placeholder since the exact structure
    of ground_truth_df depends on the dataset.
    """
    print("Blocking evaluation requires specific ground truth formatting.")
    print(f"Total candidates generated: {len(candidates_df)}")
    # In a real scenario, we'd join ground_truth_df with candidates_df
    # to see what percentage of true matches are present in candidates_df.
