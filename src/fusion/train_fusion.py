# src/fusion/train_fusion.py
import pandas as pd, numpy as np, joblib, pathlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]

def main(df_path="data/fusion_df.csv"):
    df = pd.read_csv(ROOT/df_path)
    X = df.drop(columns=["File","y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42)

    pipe = ImbPipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("gbm",   GradientBoostingClassifier())
    ])

    params = {"gbm__n_estimators":[150,250],
              "gbm__learning_rate":[0.05,0.1],
              "gbm__max_depth":[2,3]}
    grid = GridSearchCV(pipe, params, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    print("Test ROC-AUC:", round(auc,3))
    print(classification_report(y_test, y_pred))

    joblib.dump(grid.best_estimator_, ROOT/"models/fusion_gbm.pkl")
    print("Saved models/fusion_gbm.pkl")

if __name__ == "__main__":
    main()
