import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

plt.rcParams["font.size"] = 14


def main():
    score_data_paths = glob.glob("../score_data/*.csv")
    all_score_df = pd.concat([preprocess(path) for path in score_data_paths])
    X, y = make_Xy(all_score_df)
    predict_and_eval(X, y)


def preprocess(path):
    score_df = pd.read_csv(path)
    # score_df["num_of_racer"] = 
    score_df["score_sum"] = score_df["score_before"].sum()
    score_df["score_mean"] = score_df["score_before"].mean()
    score_df["score_std"] = score_df["score_before"].std()
    score_df["score_median"] = score_df["score_before"].median()
    score_df["score_min"] = score_df["score_before"].min()
    score_df["score_max"] = score_df["score_before"].max()

    for stat in ("mean", "std", "median", "min", "max"):
        score_df[f"diff_from_{stat}"] = score_df["score_before"] - score_df[f"score_{stat}"]

    score_df = score_df.drop(columns="score_after")

    return score_df


def make_Xy(data):
    X = data.drop(columns="rate_fluc")
    y = data["rate_fluc"]

    le = LabelEncoder()
    X["rank"] = le.fit_transform(X["rank"])
    
    return X, y


def predict_and_eval(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        "objective": "regression",
        "learning_rate": 0.01,
        "max_depth": 6,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_val,
        num_boost_round=1000,
        callbacks=[log_evaluation(-1), early_stopping(10)]
    )

    y_val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"valid rmse: {rmse}")
    save_feature_importance(model, X.columns)


def save_feature_importance(model, columns):
    importance_df = pd.DataFrame({"importance": model.feature_importance(importance_type="gain"), "feature": columns})
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="importance", y="feature", data=importance_df[:10], orient="h")
    plt.title(f"Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.savefig("../output/feature_importance.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
