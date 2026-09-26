# Tech Stack and Approach

## Chosen Tech Stack

*   **Language:** Python 3.10+
*   **Data Processing:** `pandas`, `numpy`, `scipy` (for sparse matrix operations during blocking)
*   **String Matching / Fuzzy Logic:** `RapidFuzz` (MIT License) - significantly faster than `thefuzz` for Levenshtein/Jaro-Winkler calculations.
*   **Feature Engineering (NLP):** `scikit-learn` (for TF-IDF and MinMax scaling)
*   **Semantic Model (Optional Features):** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0 License, 22M parameters) - highly optimized for sentence similarity.
*   **Matching Model:** `lightgbm` (MIT License) - fast, efficient gradient boosting framework tailored for tabular features.

All libraries and models strictly adhere to open-source licenses (MIT/Apache-2.0) and are well under the 8B parameter limit.

## Architecture Pipeline

```mermaid
flowchart TD
    subgraph Data Ingestion & Preprocessing
        A1[Source 1 Data] --> B[Text Normalization]
        A2[Source 2 & 3 Data] --> B
        B --> B1[Lowercase, clean punctuation, collapse whitespace]
        B1 --> B2[Frequency-based common term down-weighting]
    end

    subgraph Blocking (Candidate Generation)
        B2 --> C1[Character Trigram TF-IDF Vectorization]
        C1 --> C2[Sparse Matrix Multiplication]
        C2 --> C3[Top-K Candidate Selection Thresholding]
        C3 --> C4[(candidate_pairs.tsv)]
    end

    subgraph Feature Engineering
        C4 --> D1[Extract String Distances: Levenshtein, Jaro-Winkler]
        C4 --> D2[Extract Token Features: Token Sort/Set Ratios]
        C4 --> D3[Extract Semantic Features: all-MiniLM-L6-v2 embeddings]
        D1 --> D4[Combine into Tabular Feature Matrix]
        D2 --> D4
        D3 --> D4
    end

    subgraph Matching Model & Post-processing
        D4 --> E1[LightGBM Binary Classifier]
        E1 --> E2[Probability Predictions]
        E2 --> E3[Apply Threshold Tuned for F_0.5]
        E3 --> E4{Matches found for Source 1?}
        E4 -- Yes --> F1[Output Matches]
        E4 -- No --> F2[Implicit Singleton]
        F1 --> G[(matching_results.tsv)]
        F2 --> G
    end
```

## Rationale Summary for Major Design Decisions

1.  **Character N-grams for Blocking:** Instead of semantic blocking, character n-grams are robust to severe typos and handle arbitrary languages/regions (like the hidden French test set) natively without hardcoded rules.
2.  **Sparse Matrix Multiplication over Approximate Nearest Neighbors (ANN):** Given the relatively modest scale of the provided dataset vs billions, exact sparse matrix multiplication of TF-IDF vectors is extremely fast and provides perfect cosine similarity scores, removing the hyperparameter tuning complexities of ANN (like HNSW or FAISS) while guaranteeing high recall.
3.  **LightGBM over Deep Learning Matching:** With engineered string distances and TF-IDF features, tabular models like LightGBM consistently outperform deep neural networks in both speed and accuracy. It allows for extremely fast iteration during threshold tuning.
4.  **Frequency-based Cleaning vs Hardcoded Dictionaries:** To avoid overfitting to US/IN data, we will not hardcode a list of terms like "LLC" or "Pvt". Instead, we detect high-frequency terms in the corpus computationally and down-weight them, naturally handling French equivalents like "SARL" without prior knowledge.
5.  **Threshold Tuning for F_0.5:** The F_0.5 metric heavily penalizes false positives and rewards true negatives (singletons). A standard 0.5 probability threshold will fail terribly. We treat the classification threshold as a hyperparameter and optimize it directly against the metric on a held-out validation set.

## Glossary of Explicitly Handled Noise Patterns

*   **Typographical Errors (Typos):** Handled by Levenshtein distance and character n-gram blocking.
*   **Abbreviation vs. Full Word (e.g., "Mfg" vs "Manufacturing"):** Handled by token-set ratio features and (partially) by sentence embeddings.
*   **Word Reordering (e.g., "Technologies Acme" vs "Acme Technologies"):** Handled by Token Sort Ratio feature.
*   **Missing Components (e.g., missing suite number or state):** Handled by Token Set Ratio and Jaccard similarity, which don't penalize missing subsets as heavily as exact matching.
*   **Frequent Legal Suffixes (e.g., "LLC", "Ltd", "Inc", "SA"):** Handled by frequency-based term down-weighting during TF-IDF, preventing two unrelated "LLC"s from matching solely based on the suffix.
*   **Inconsistent Case and Punctuation (e.g., "O'Reilly", "O Reilly", "OREILLY"):** Handled by aggressive lowercasing and punctuation stripping during initial preprocessing.
