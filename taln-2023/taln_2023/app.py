import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def f1_ratio_figure(
    csv_filename: str = "wandb_export_2023-02-13T17_59_44.642+01_00.csv",
    ax=None,
):
    df = pd.read_csv(f"../assets/{csv_filename}")
    if "data/fold" in df.keys():
        sns.scatterplot(
            x="data/ratio",
            y="test/tokens/MulticlassF1Score",
            data=df,
            hue=df["data/fold"],
            ax=ax,
        )
    else:
        sns.scatterplot(
            x="data/ratio",
            y="test/tokens/MulticlassF1Score",
            color="salmon",
            data=df,
            ax=ax,
        )


def unfolded_ratio_figure(
    csv_filename: str = "wandb_export_2023-02-13T18_57_26.877+01_00.csv",
    ax=None,
):
    f1_ratio_figure(csv_filename, ax)


def scatter_plot_figures():
    sns.set_style("whitegrid")
    sns.set_context("paper")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    f1_ratio_figure(ax=ax[0])
    unfolded_ratio_figure(ax=ax[1])
    ax[0].set_xticks(np.arange(0, 1.1, 0.1))
    ax[0].set_yticks(np.arange(0.65, 0.78, 0.01))
    ax[0].set_xlabel("Data Ratio")
    ax[0].set_ylabel("Aggregate F1 Score")
    ax[1].set_xticks(np.arange(0, 1.1, 0.1))
    ax[1].set_yticks(np.arange(0.65, 0.78, 0.01))
    ax[1].set_xlabel("Data Ratio")
    ax[1].set_ylabel("Aggregate F1 Score")
    plt.tight_layout()
    plt.savefig("../pdf/scatters.pdf")


if __name__ == "__main__":
    scatter_plot_figures()
