import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def visualise_gt_noised_and_predicted(noised, gt, predicted, epoch, path: str = None):
    plt.figure(figsize=(20, 10))
    plt.plot(noised[0], label="noised")
    plt.plot(gt[0], label="gt")
    plt.plot(predicted[0], label="predicted")
    plt.legend()
    if path:
        plt.savefig(f"{path}/epoch_{epoch}.png")
    plt.close()
    # plt.show()
