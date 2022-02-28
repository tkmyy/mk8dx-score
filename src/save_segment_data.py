import os
import glob
import cv2
import matplotlib.pyplot as plt


def main():
    filepaths = glob.glob("../images/*.PNG")
    for filepath in filepaths:
        save_segment_data(filepath)


def save_segment_data(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    img_bgr = cv2.imread(filepath)
    img_gray = color_to_gray(img_bgr, thr=150)

    row_length = 12
    w_ranges = [
        [1050, 1080],
        [1080, 1093],
        [1093, 1105],
    ] + \
    [
        [1127+i*18, 1127+(i+1)*18]
        for i in range(5)
    ]

    for row in range(row_length):
        is_white_back = False
        row_data = img_gray[52*(row+1)+10:51+52*(row+1)-10, :, :]
        if (row_data[:, 1050:1225, :].flatten() > 200).sum() > 10000:
            is_white_back = True

        for idx, (w_min, w_max) in enumerate(w_ranges):
            seg_data = row_data[:, w_min:w_max, :]
            if is_white_back:
                seg_data = cv2.bitwise_not(seg_data)
            seg_data = cv2.resize(seg_data, (30, 60))
            cv2.imwrite(f"../7seg_datasets/{basename}_{row}_{idx}.jpg", seg_data)


def color_to_gray(img, thr):
    img_bgr_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bgr_gray = cv2.threshold(img_bgr_gray, thr, 255, cv2.THRESH_BINARY)
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_gray = cv2.cvtColor(img_bgr_gray, cv2.COLOR_BGR2RGB)

    return img_rgb_gray


if __name__ == "__main__":
    main()
