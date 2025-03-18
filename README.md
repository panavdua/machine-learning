**Multi Modal Classification Model**

This project implements a classification model that processes both tabular and image data using RandomForestClassifier. The model performs extensive feature engineering, dataset balancing, and preprocessing to improve classification accuracy.

**Features**
- Handles Missing Values: Fills NaNs using mode-based imputation.
- Encodes Categorical Features: Converts string-based categorical values (e.g., "C123" → 123").
- Removes Highly Correlated Features: Drops features with correlation > 0.95.
- Scales Numerical Data: Uses StandardScaler for normalization.
- Processes Image Data: Flattens image arrays and incorporates them as additional tabular features.
- Balances Dataset: Uses oversampling to handle class imbalance.
- Trains a RandomForestClassifier with optimized hyperparameters for classification.
