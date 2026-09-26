# Amazon ML Challenge 2026 - Business Entity Resolution

This repository contains the solution for the Business Entity Resolution track.

## Setup

1.  **Extract Data:** Ensure the dataset is extracted into `data/dataset/train/`. The required files are:
    *   `train_source1.tsv`
    *   `train_source2.tsv`
    *   `train_source3.tsv`
    *   `train_ground_truth.tsv`

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipeline

To run the end-to-end pipeline (Preprocessing -> Blocking -> Feature Engineering -> Matching):

```bash
python run_pipeline.py --mode train
```

This will generate two files in the `output/` directory:
*   `candidate_pairs.tsv`: The output of the blocking stage.
*   `matching_results.tsv`: The final predictions.

## Validating Submission

Once the pipeline finishes, run the validation script to ensure formatting is correct:

```bash
python utils/validate_submission.py
```
*(Note: `utils/validate_submission.py` should be provided by the challenge organizers)*

## Methodology Details

Please refer to `TECH_STACK_AND_APPROACH.md` for a comprehensive breakdown of the architecture, design choices, and how we optimize for the F_0.5 metric.
