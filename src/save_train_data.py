import argparse

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model


def main(basename):
    lr = 0.001
    drop_rate = 0.4
    model = load_model(f"../models/cnn_lr-{lr}_drop-{drop_rate}.h5")

    X = make_X(basename)
    res = pd.DataFrame(model.predict_classes(X).reshape(12, 8))
    score_df = format_results(res)
    score_df.to_csv(f"../score_data/{basename}.csv", index=None)


def make_X(basename):
    X = []
    for row in range(12):
        for idx in range(8):
            data = cv2.imread(f"../7seg_datasets/{basename}_{row}_{idx}.jpg")[:, :, 0]
            X.append(data)

    X = np.array(X)
    X = X.astype("float32")
    X = X / 255.0
    X = X.reshape(-1, 60, 30, 1)

    return X


def format_results(res):
    res = res.astype(str)
    res = res.replace({"10": ""})
    score = res.apply(lambda x: "".join(x[i] for i in range(3, 8)), axis=1).astype(int)
    sign = res[0].apply(lambda x: 1 if x == "11" else -1)
    fluc = (res[1] + res[2]).astype(int)
    fluc = sign * fluc

    score_df = pd.concat([fluc, score], axis=1)
    score_df = score_df.rename(columns={0: "rate_fluc", 1: "score_after"})
    score_df["rank"] = [i + 1 for i in range(12)]
    score_df["score_before"] = score_df["score_after"] - score_df["rate_fluc"]

    score_df = score_df.reindex(
        columns=["rank", "rate_fluc", "score_after", "score_before"]
    )

    return score_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basename",
        type=str,
    )
    args = parser.parse_args()
    basename = args.basename
    main(basename=basename)
