import matplotlib.pyplot as plt
import numpy as np

def visualize_corr(corr,labels):
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def visualize_pie_char(data, labels_name):
    values, counts = np.unique(data[labels_name], return_counts=True)
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=values, autopct='%1.1f%%');
    plt.show()