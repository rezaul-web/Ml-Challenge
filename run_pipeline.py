import argparse
import yaml
from src.candidate_generation import run_candidate_generation
from src.matching_pipeline import run_matching_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--phase", choices=["all", "1", "2"], default="all", 
                        help="Phase to run: 1 (Blocking), 2 (Matching), or all")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.phase in ["all", "1"]:
        print("========== RUNNING PHASE 1: CANDIDATE GENERATION ==========")
        success = run_candidate_generation(config)
        if not success:
            return
            
    if args.phase in ["all", "2"]:
        print("\n========== RUNNING PHASE 2: MATCHING PIPELINE ==========")
        run_matching_pipeline(config)

if __name__ == "__main__":
    main()
