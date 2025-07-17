# 📣 Bank Term Deposit Campaign Response Prediction  
## A Comparative Study Using Logistic Regression, Decision Tree, and Random Forest

This project analyzes customer responses to a bank's term deposit marketing campaign. Using historical data and Python, we developed and evaluated response models to predict whether a customer will subscribe to a term deposit. The goal is to identify the most effective classification method for improving campaign targeting and maximizing marketing performance.

---

## 🧠 Models Compared
- **Logistic Regression** – Interpretable baseline model
- **Decision Tree** – Tree-based structure for rule-based segmentation
- **Random Forest** – Ensemble model for better generalization and accuracy

---

## 🔍 Key Findings

### 📊 Performance Summary

| Metric              | Logistic Regression | Decision Tree | Random Forest |
|---------------------|---------------------|----------------|----------------|
| **AUC**             | 0.87                | 0.86           | **0.87**       |
| **Accuracy**        | **88.33%**          | 87.55%         | 86.98%         |
| **Sensitivity**     | 40.84%              | 44.87%         | **55.46%**     |
| **Specificity**     | **94.62%**          | 93.20%         | 91.15%         |
| **F1-Score**        | 45.02%              | 46.00%         | **49.90%**     |
| **True Positives**  | 486                 | 534            | **660**        |
| **False Negatives** | 704                 | 656            | **530**        |

✅ **Random Forest** emerged as the top model due to:
- The **highest sensitivity**, critical in capturing potential responders
- A **balanced F1-score**, combining precision and recall
- The **lowest false negatives**, minimizing missed opportunities

---

## 🛠️ Tools & Libraries Used
- Python, Jupyter Notebook  
- `pandas`, `numpy`, `seaborn`, `matplotlib`
- `scikit-learn`
- `scikit-plot` for ROC & gain curves

---

## 📁 Repository Structure
bank-response-prediction/
1. data/ # Cleaned dataset
2. notebooks/ # Model development and EDA
3. results/ # Evaluation plots and metrics
4. README.md # Project documentation

---

## 📌 Conclusion

This study demonstrates that while **Logistic Regression** offers high interpretability and accuracy, **Random Forest** provides the best trade-off between recall and precision, making it the most suitable model for campaign optimization.

Organizations aiming to **maximize subscription rates** should use Random Forest to **prioritize outreach**, ensuring minimal missed opportunities and effective resource allocation.

---
