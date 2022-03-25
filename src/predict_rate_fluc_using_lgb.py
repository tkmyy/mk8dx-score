import glob
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

plt.rcParams["font.size"] = 14


def main():
    score_data_paths = glob.glob("../score_data/*.csv")
    all_score_df = pd.concat([preprocess(path) for path in score_data_paths])
    predict_and_eval(all_score_df)


def preprocess(path):
    score_df = pd.read_csv(path)
    score_df["score_sum"] = score_df["score_before"].sum()
    score_df["score_mean"] = score_df["score_before"].mean()
    score_df["score_std"] = score_df["score_before"].std()
    score_df["score_median"] = score_df["score_before"].median()
    score_df["score_min"] = score_df["score_before"].min()
    score_df["score_max"] = score_df["score_before"].max()

    for stat in ("mean", "std", "median", "min", "max"):
        score_df[f"diff_from_{stat}"] = (
            score_df["score_before"] - score_df[f"score_{stat}"]
        )

    score_df = score_df.drop(columns="score_after")
    score_df["filename"] = os.path.splitext(os.path.basename(path))[0]

    return score_df


def predict_and_eval(data):
    filename_list = data["filename"].unique()
    params = {
        "objective": "regression",
        "learning_rate": 0.01,
        "max_depth": 6,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_filename_idx, test_filename_idx in kf.split(filename_list):
        train_filename = filename_list[train_filename_idx]
        test_filename = filename_list[test_filename_idx]

        train_data = data[data["filename"].isin(train_filename)]
        test_data = data[data["filename"].isin(test_filename)]

        X_train = train_data.drop(columns=["filename", "rate_fluc"])
        y_train = train_data["rate_fluc"]
        X_test = test_data.drop(columns=["filename", "rate_fluc"])
        y_test = test_data["rate_fluc"]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_test, y_test, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_val,
            num_boost_round=5000,
            callbacks=[log_evaluation(-1), early_stopping(10)],
        )

        y_test_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        print(f"valid rmse: {rmse}")
        # save_feature_importance(model, X_train.columns)


def save_feature_importance(model, columns):
    importance_df = pd.DataFrame(
        {
            "importance": model.feature_importance(importance_type="gain"),
            "feature": columns,
        }
    )
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x="importance", y="feature", data=importance_df[:10], orient="h")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    # plt.savefig("../output/feature_importance.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
