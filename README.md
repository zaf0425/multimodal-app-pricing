# Price Tier Classification of Mobile Applications Across Apple App Store and Google Play

Can the price tier of a paid mobile app be predicted from its public metadata, and does a price pattern learned on one marketplace transfer to the other? This project classifies paid apps into **Cheap**, **Mid** and **Expensive** tiers using numerical metadata, app-title text, and a fusion of both, under same-platform and cross-platform settings. SHAP and LIME are used to explain what the models learned.

*Course project for CSE424: Pattern Recognition, BRAC University.*

## Research Questions

1. How do numerical, textual, and combined (multimodal) features compare for price-tier prediction?
2. Does fusing numerical and textual features improve on text alone?
3. How well does a model trained on one store generalize to the other (domain shift)?
4. Which features and title words drive the predictions (explainability)?

## Datasets

| | Apple App Store | Google Play |
|---|---|---|
| Raw listings | 1,230,376 apps (20 columns) | not stated in report |
| Paid listings | 102,992 (8.4%) | 44,933 |
| Final cleaned set | **57,082 apps** | **16,338 apps** |

Final tier distribution (Cheap / Mid / Expensive):

- Apple: 58.7% / 26.7% / 14.6%
- Google: 52.9% / 33.6% / 13.5%

**Price tiers:** Cheap (< $3), Mid ($3 to < $9), Expensive (≥ $9). All prices are in USD.

The notebook expects two CSVs, `AppleData.csv` and `GoogleData.csv` (the report refers to them as the Apple App Store and Google-Playstore listing files). They are not bundled in this repo.

## Pipeline

1. **Filtering and labeling:** remove free apps and non-positive prices, convert price strings to numbers, build the three price tiers.
2. **Cleaning and harmonization:** drop high-null columns (e.g. developer website), drop remaining nulls and duplicates, standardize column names, map both stores' genres into a common taxonomy.
3. **Feature engineering**
   - *Numerical:* Rating, Rating Count, Size (MB) and similar fields, scaled with `StandardScaler` (fit on training data only). Same-platform runs also use store-specific fields such as install counts.
   - *Textual (app titles only):* ASCII-normalized, lowercased, cleaned; represented as TF-IDF (5,000 uni/bigrams), Word2Vec (100-d, mean pooled), and character-level sequences (length 30).
   - *Multimodal:* concatenation of scaled numerical and TF-IDF features for classical models; dual-branch **BiLSTM-Fusion** and **Transformer-Fusion** networks for deep models.
4. **Experiments:** same-platform (80/20 stratified split, seed 42) and cross-platform (train on one store, test on the other) for each modality.
5. **Tuning and evaluation:** manual hyperparameter tuning, with **macro-F1** as the primary metric because of class imbalance.
6. **Explainability:** SHAP (global and local) on the numerical Logistic Regression models and the best textual model, and LIME for individual title predictions.

### Models

Logistic Regression, Linear SVM, Random Forest, XGBoost, LightGBM, Multinomial and Complement Naive Bayes, MLP, BiLSTM, GRU, and BiLSTM/Transformer fusion networks.

## Results

Best model per setting (Train → Test):

| Features | Apple → Apple | Google → Google | Apple → Google | Google → Apple |
|---|---|---|---|---|
| **Numerical** (macro-F1) | 0.37 | 0.37 | 0.27 | 0.32 |
| **Textual** (macro-F1) | 0.53 | 0.57 | 0.43 | 0.41 |
| **Multimodal** (macro-F1) | 0.52 | 0.57 | 0.44 | 0.40 |
| Multimodal accuracy | 0.67 | 0.61 | 0.47 | 0.56 |

**Key findings**

- **Text beats numbers.** App titles outperform numerical metadata in every setting. Words such as "pro", "premium" and "plus" carry strong price signals.
- **Numerical-only models mostly fall back on the majority "Cheap" class.** Apple numerical accuracy is 0.53 but macro-F1 is only 0.37.
- **Fusion helps accuracy, not macro-F1.** Multimodal models reach the best same-platform accuracy (0.67 on Apple) and precision, but do not consistently improve macro-F1 over text alone.
- **Cross-platform transfer is partial.** Macro-F1 drops from 0.53–0.57 within a platform to 0.41–0.43 across platforms, which reflects differences in title conventions and pricing between the two stores.
- **A well-regularized linear model on good features** was often as good as more complex models for this task.

## Repository Structure

```
.
├── PROJECT_WITH_EXPLAINABLE_AI.ipynb   # Full pipeline: cleaning, EDA, models, SHAP/LIME
├── Project_Proposal.pdf
├── Project_Report.pdf                  # Full paper-style report
├── Project_Presentation.pdf            # Course presentation slides
├── AppleData.csv                       # (add yourself, not included)
└── GoogleData.csv                      # (add yourself, not included)
```

## Getting Started

The notebook was written for **Google Colab** and loads the CSVs from Google Drive (`/content/drive/MyDrive/`). To run it locally, change the two `pd.read_csv(...)` paths in the first cells and remove the `drive.mount` call.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn gensim xgboost lightgbm \
            tensorflow shap lime jupyter
jupyter notebook PROJECT_WITH_EXPLAINABLE_AI.ipynb
```

## Authors

- Zafra Musarrat Islam
- Karib Fiyaz
- Tayassuk Md Abir
- Sadman Zarif Kazmi

Department of Computer Science, BRAC University. Instructor: Annajiat Alim Rasel.

## License

Add a license of your choice (e.g. MIT) before publishing.
