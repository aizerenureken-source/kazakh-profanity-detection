# Kazakh Profanity Detection

This DSSC5113 project classifies review text for Kazakh profanity. The model accepts one review and returns a binary label and a score. Label `1` indicates Kazakh profanity, and label `0` indicates its absence.

## Dataset and labeling

The reviews come from [KazSAnDRA](https://github.com/IS2AI/KazSAnDRA). A classmate shared the initial selection and labels. I reviewed and relabeled the selected examples for this project. The dataset includes product feedback and reviews of apps and games. No synthetic examples were added.

Recognizable masked spellings, transliterations, and spelling variants of Kazakh profanity count as class `1`. Negative feedback, ordinary insults count as class `0`. Mixed-language reviews are labeled according to whether they contain Kazakh profanity. Context is used to resolve ambiguous spellings.

| Example | Label | Reason |
|---|---:|---|
| «ал кеттик» | 0 | Neutral phrase |
| «қосылмайды ғой мал» | 1 | The word «мал» is used as an insult in this context. |
| «қысқасаын айтқанда к*т приложения екен дұрыс істемейді» | 1 | Recognizable masked Kazakh profanity |

One duplicate, one unlabeled row, and four rows containing numbers in place of review text were excluded. Identified near-duplicates were assigned to the same split. The splits were created with seed `42` and are recorded in the `split` column of `kazakh_profanity_2200-8.xlsx`. The counts below reflect the final notebook run.

| Split | Class 0 | Class 1 | Total |
|---|---:|---:|---:|
| Training | 728 | 806 | 1,534 |
| Validation | 159 | 171 | 330 |
| Test | 155 | 175 | 330 |
| **Total** | **1,043** | **1,151** | **2,194** |

## Model and training

Text is normalized with Unicode NFKC, case folding, and whitespace normalization. Punctuation is retained so that some masking patterns can be learned. The full review is processed, and unseen n-grams are ignored.

Character TF-IDF uses n-grams of length 2–5, `min_df=2`, and sublinear term frequency. Character features allow different spellings and inflected forms to share features. Logistic regression uses L2 regularization, `C=1.0`, the `lbfgs` solver, and up to 1,000 iterations. Both the vectorizer and the classifier are fitted on the training set.

Thresholds from 0.20 to 0.80 are evaluated in steps of 0.01 on the validation set. The threshold with the highest class-1 F1 is selected. Ties are resolved in favor of the threshold closest to 0.50. The selected threshold was **0.46**, with a validation F1 of **0.914**.

## Run in Google Colab

1. Open `kazakh_profanity.ipynb` in Colab and select a CPU runtime.
2. Upload `kazakh_profanity_2200-9.xlsx` to the working directory. The filename must match the `DATA` path in the notebook.
3. Run the cells from top to bottom. The setup cell installs the pinned training dependencies. The notebook writes `metrics.json` and `model.joblib`.

To predict without retraining, upload the saved `model.joblib`, run the setup cell, and run the last code cell. Replace the example text with any review. The saved file contains the fitted vectorizer, classifier, and decision threshold.

## Test results

On 330 test reviews, class-1 **precision was 88.8%**, **recall was 91.4%**, and **F1 was 90.1%**.

| True label | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 | 135 | 20 |
| 1 | 15 | 160 |

False positives can hide legitimate feedback. False negatives allow profanity to remain visible.

## Error analysis

- **Neutral phrase flagged (`pla034591pla`, score 0.642).** The short phrase «ал кеттик» was predicted as class 1. Short character sequences shared with profane training examples may have influenced the score. Adding more short, neutral training examples could help.
- **Masked profanity missed (`pla030335pla`, score 0.362).** The model predicted class 0 for a review containing «к*т». The asterisk changes the character n-grams, which can make the masked spelling harder to recognize. Adding more masked variants to the training set could improve recall.
- **Inconsistent label for an insult (`pla071821pla`, score 0.435).** The model assigned class 0 to «қосылмайды ғой мал», while the recorded label was 1. The word «мал» is used as an insult here.

## Limitations

Some test labels were revised after prediction errors were inspected. The reported metrics use the revised labels. A fresh set annotated before reviewing model predictions would provide an independent evaluation. The test set includes app and game reviews, so performance on physical product reviews was not assessed separately. The proportion of profanity in a live stream of reviews may differ from that in this dataset.

## Contributions and AI assistance

I reviewed and corrected labels, ran the notebook in Colab, and inspected prediction errors. AI assistance was used to prepare the data splits, and explain results.