import os
import joblib
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def load_data(path='data/credit_data.csv'):
    df = pd.read_csv(path)
    return df

def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    base_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    calib = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=3)

    pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', calib)
    ])

    return pipe

def get_feature_names(preprocessor, num_cols, cat_cols):
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = list(num_cols) + cat_feature_names
    return feature_names

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    df = load_data()
    target = 'default'

    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols = [c for c in num_cols if c != target]  # remove target if numeric
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    X = df[num_cols + cat_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(num_cols, cat_cols)

    print("Training pipeline... (this can take a little while)")
    pipeline.fit(X_train, y_train)

    probs_test = pipeline.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs_test)
    print(f"Test ROC-AUC: {auc:.4f}")

    preds = (probs_test >= 0.5).astype(int)
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(pipeline, 'models/credit_pipeline.pkl')
    print("Saved trained pipeline to models/credit_pipeline.pkl")
    preprocessor = pipeline.named_steps['pre']
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("Saved feature names to models/feature_names.pkl (len=", len(feature_names), ")")
