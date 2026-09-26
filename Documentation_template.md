# Documentation Template

## 1. Team Details
*   Team Name: [Insert Name]
*   Members: [Insert Names]

## 2. Approach Summary
We tackle the entity resolution problem using a two-stage approach:
1.  **Blocking:** Character trigram TF-IDF vectorization with sparse matrix dot product to identify top-K candidates efficiently, maximizing recall.
2.  **Matching:** A LightGBM classifier trained on multiple string distance metrics (Levenshtein, Jaro-Winkler, Token Ratios). The probability threshold is explicitly tuned on a held-out validation set to maximize the competition's macro-averaged F_0.5 metric.

## 3. Handling Unseen Geographies (France)
Our pipeline relies strictly on string-level and character-level similarities rather than hardcoded geographical dictionaries (e.g., US states) or language-specific rules. Term down-weighting is done statistically via TF-IDF, allowing the pipeline to natively adapt to French business entity names and addresses.

## 4. Model Details
*   Model Name: LightGBM
*   Parameters: ≤ 1M (well within 8B limit)
*   License: MIT
