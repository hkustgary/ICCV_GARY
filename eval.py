import numpy as np
from sklearn.metrics import f1_score
from skimage.transform import resize
import os
import csv


def compute_f1_score(mask1, mask2):
    """
    Computes the F1 score between two segmentation masks, resizing one to match the size of the other.
    
    Parameters:
    - mask1: np.array, first binary segmentation mask
    - mask2: np.array, second binary segmentation mask
    
    Returns:
    - f1: float, the F1 score between the two resized masks
    """
    # Resize mask2 to match the shape of mask1
    mask2_resized = resize(mask2, mask1.shape, mode='constant', anti_aliasing=False, preserve_range=True).astype(np.uint8)
    # print(mask2_resized)
    # mask2_resized = (mask2_resized > 0.5).astype(np.uint8)  # Thresholding to convert back to binary mask

    # Flatten the masks to 1D arrays
    mask1_flat = mask1.flatten().astype(np.uint8)
    mask2_resized_flat = mask2_resized.flatten()
 

    # Calculate F1 score
    f1 = f1_score(mask1_flat, mask2_resized_flat)
    
    return f1

# # Example masks
# predict = np.load('predict/test_001_predict.npy')

# gt = np.load('gt/test_001.npy')
# # Compute the F1 score
# f1 = compute_f1_score(predict, gt)
# Directory paths
predict_dirs = {'predict/dsmil/ori','predict/dsmil/prolif','predict/dsmil/prolif_cor'}
gt_dir = 'gt/224_1'

# Initialize counters for TN, TP, FN, FP

import matplotlib.pyplot as plt
# Iterate over all files in the predict directory

file_name = []
f1_ori = []
f1_pro = []
f1_pro_cor = []
for predict_dir in predict_dirs:
    count = 0
    TN, TP, FN, FP, F1 = 0, 0, 0, 0, 0
    for predict_file in os.listdir(predict_dir):
        if predict_file.endswith('.npy'):
            # Load the prediction and ground truth masks
            predict = np.load(os.path.join(predict_dir, predict_file))
            gt = np.load(os.path.join(gt_dir, predict_file.replace('_predict', '_patch')))

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(predict, cmap='gray')
            axs[0].set_title('Prediction')
            axs[1].imshow(gt, cmap='gray')
            axs[1].set_title('Ground Truth')
            plt.tight_layout()
            plt.savefig(os.path.join(predict_dir, predict_file.replace('.npy', '.png')))
            plt.close()

            # Resize and flatten masks
            mask2_resized = resize(gt, predict.shape, mode='constant', anti_aliasing=False, preserve_range=True).astype(np.uint8)
            predict_flat = predict.flatten().astype(np.uint8)
            gt_flat = mask2_resized.flatten()
            f1 = compute_f1_score(predict_flat, gt_flat)
            # Calculate TN, TP, FN, FP
            if predict_dir == 'predict/dsmil/ori':
                f1_ori.append(f1)
                file_name.append(predict_file)
            elif predict_dir == 'predict/dsmil/prolif':
                f1_pro.append(f1)
            else:
                f1_pro_cor.append(f1)
            TN += np.sum((predict_flat == 0) & (gt_flat == 0))
            TP += np.sum((predict_flat == 1) & (gt_flat == 1))
            FN += np.sum((predict_flat == 0) & (gt_flat == 1))
            FP += np.sum((predict_flat == 1) & (gt_flat == 0))
            F1 += f1
            count += 1

    # Print the results
    print("Predictions:", predict_dir)
    print("True Negatives (TN):", TN)
    print("True Positives (TP):", TP)
    print("False Negatives (FN):", FN)
    print("False Positives (FP):", FP)
    print(F1/count)
with open('/home/mdi/WS-FSS-Code/data/predict_eval.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "f1_ori", "f1_pro", "f1_pro_cor"])
    for i in range(len(file_name)):
        writer.writerow([file_name[i], str(f1_ori[i]), str(f1_pro[i]), str(f1_pro_cor[i])])
# print("F1 Score:", f1)