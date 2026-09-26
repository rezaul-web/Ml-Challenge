---

## Problem Statement

### 📢 Important Update

> **`candidate_pairs.tsv` is part of your final submission.**

1. **Blocking has to scale.** Amazon resolves business entities across billions of records, so comparing every record with every other one is not an option.  
   Your blocking / candidate-generation step must cut the search space to a small candidate set per Source 1 entity.

2. **Candidate generation counts toward the final ranking.** We will review your `candidate_pairs.tsv` and the code that produces it when deciding final rankings, alongside your `matching_results.tsv` score. The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard.

---

# Business Entity Resolution Challenge

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities.

These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as **Entity Resolution (ER)**.

Your challenge is to build an ML solution that, given business records from **3 independent data sources** with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

### Source 1

Source 1 is the deduplicated reference source.

Your task is to find **all matching records from Source 2 and Source 3 for each Source 1 entity**.

A Source 1 entity may match:

- Zero records from Source 2 and Source 3
- One record
- Multiple records

---

## Dataset

**Download Data Set:** Click here to Download

---

# File Format

All files in this challenge are **tab-separated (`.tsv`)**, and your submissions must be tab-separated too.

Tabs are used because business addresses and the ID list columns both contain commas.

Read them with an explicit tab separator:

```python
import pandas as pd

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")