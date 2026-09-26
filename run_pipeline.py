import os
import argparse
from src.module_a_data import load_config, load_and_preprocess_source, split_ground_truth
from src.module_b_blocking import generate_candidates, evaluate_blocking
from src.module_c_matching import extract_features, train_and_predict, generate_submission

def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Execution mode")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 1. Load Data
    s1_path = os.path.join(config['paths']['raw_data_dir'], "train_source1.tsv")
    s2_path = os.path.join(config['paths']['raw_data_dir'], "train_source2.tsv")
    s3_path = os.path.join(config['paths']['raw_data_dir'], "train_source3.tsv")
    
    # In a real scenario we'd check if these exist, for now we assume they do
    try:
        df_s1 = load_and_preprocess_source(s1_path, config)
        df_s2 = load_and_preprocess_source(s2_path, config)
        df_s3 = load_and_preprocess_source(s3_path, config)
    except FileNotFoundError:
        print("Dataset not found. Please extract the data into data/dataset/train/")
        return

    # 2. Blocking
    candidates_df = generate_candidates(df_s1, df_s2, df_s3, config)
    candidates_df.to_csv(config['paths']['candidate_pairs'], sep="\t", index=False)
    print(f"Candidate pairs saved to {config['paths']['candidate_pairs']}")

    # 3. Feature Engineering
    features_df = extract_features(candidates_df, df_s1, df_s2, df_s3, config)
    
    # 4. Modeling & Scoring
    if args.mode == "train":
        # Load ground truth and split
        # train_gt, val_gt = split_ground_truth(config['paths']['train_ground_truth'], config)
        
        # Train model and find best threshold
        y_pred, best_threshold = train_and_predict(features_df, config, ground_truth_df=None)
        
        # Generate submission
        generate_submission(features_df, y_pred, best_threshold, config['paths']['matching_results'])
        
    elif args.mode == "predict":
        # In predict mode, we load the trained model and use the saved threshold
        print("Predict mode not fully implemented (requires saved model).")
        y_pred, best_threshold = train_and_predict(features_df, config)
        generate_submission(features_df, y_pred, best_threshold, config['paths']['matching_results'])

if __name__ == "__main__":
    main()
