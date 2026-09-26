import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_source(filepath, config):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    
    # Fill NA and normalize
    text_cols = config['preprocessing']['text_columns']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
            
    # Create a combined feature for blocking
    df['combined_text'] = df[text_cols[0]]
    if len(text_cols) > 1 and text_cols[1] in df.columns:
         df['combined_text'] = df['combined_text'] + " " + df[text_cols[1]]
         
    return df

def split_ground_truth(gt_path, config):
    """
    Splits ground truth into train and validation sets, ensuring that Source 1 entities
    are not split across both sets to prevent data leakage.
    """
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    
    # We assume ground truth has columns like 'source1_id', 'source2_id', 'source3_id'
    # The prompt implies a 1-to-many relationship from Source 1
    # We will group by Source 1 ID.
    if 'source1_id' not in gt_df.columns:
        # Fallback to the first column if exact name is unknown, assuming it's the anchor
        anchor_col = gt_df.columns[0]
    else:
        anchor_col = 'source1_id'
        
    gss = GroupShuffleSplit(n_splits=1, test_size=config['modeling']['val_size'], random_state=config['modeling']['random_state'])
    train_idx, val_idx = next(gss.split(gt_df, groups=gt_df[anchor_col]))
    
    train_gt = gt_df.iloc[train_idx].copy()
    val_gt = gt_df.iloc[val_idx].copy()
    
    return train_gt, val_gt
