# Synapse Atlas: classifying mouse brain protein profiles

## a. Problem statement

Can protein expression measurements distinguish experimental groups in a mouse model of Down syndrome? This project compares five classic classifiers on one stratified holdout and presents an interactive scorecard for exploring their errors.

## b. Dataset description

The [UCI Mice Protein Expression dataset](https://archive.ics.uci.edu/dataset/342/mice+protein+expression) contains measurements from 1,080 mouse brain tissue samples. Seventy-seven `_N` columns record protein expression levels; the target is one of eight genotype, treatment, and behavior combinations. This niche biology dataset is licensed CC BY 4.0. Because each mouse contributes repeated measurements, the holdout is split by `MouseID` with stratified groups (five folds, seed 731), keeping each mouse on only one side. Median imputation and scaling are fitted within pipelines using training rows only.

## c. Github Repository Link

Repository: _Add your GitHub URL after creating a remote repository._

## d. Models used

Precision, recall, F1, and one-vs-rest AUC use macro averaging. MCC is the multiclass Matthews correlation coefficient. Run `python train_models.py` to create artifacts and populate `model_scores.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9954 | 0.9986 | 0.9960 | 0.9954 | 0.9956 | 0.9947 |
| Decision Tree | 0.8380 | 0.9483 | 0.8416 | 0.8409 | 0.8369 | 0.8161 |
| K-Nearest Neighbors | 0.9491 | 0.9978 | 0.9514 | 0.9500 | 0.9472 | 0.9426 |
| Gaussian Naive Bayes | 0.7685 | 0.9716 | 0.7894 | 0.7667 | 0.7676 | 0.7379 |
| Support Vector Machine | 0.9815 | 1.0000 | 0.9836 | 0.9815 | 0.9821 | 0.9790 |

### Performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best macro F1 and MCC in this holdout; scaled linear separation fits these profiles very well. |
| Decision Tree | Lowest scores; one tree's threshold rules miss some of the detailed class boundaries. |
| K-Nearest Neighbors | Strong second tier, with high macro scores but more confusion than the linear and kernel models. |
| Gaussian Naive Bayes | Weakest accuracy, likely reflecting correlated protein measurements that violate its independence assumption. |
| Support Vector Machine | Highest AUC, indicating excellent class ranking; its thresholded predictions trail logistic regression slightly. |

**Overall winner:** Logistic Regression by macro F1 (0.9956) and MCC (0.9947) on the group-held-out split. The SVM has the highest macro one-vs-rest AUC (1.0000).

## App

Run `python train_models.py`, then `streamlit run app.py`. Upload a labeled CSV containing all protein features and `class`, or choose the supplied holdout. The app loads saved models and reports accuracy, AUC, macro precision/recall/F1, MCC, a confusion map, and a per-class report. Uploaded rows are evaluated without retraining.

## Files

- `app.py` — custom Streamlit scorecard
- `train_models.py` — UCI workbook download, stratified split, training, artifact export
- `model/` — saved models and catalog created by the training script
- `test_data.csv` — labeled held-out rows created by the training script
- `model_scores.csv` — holdout comparison created by the training script
- `requirements.txt` — runtime dependencies

## Reproducibility

Install with `python -m pip install -r requirements.txt`, then run `python train_models.py` once. Seed: 731. The initial dataset download requires network access.
