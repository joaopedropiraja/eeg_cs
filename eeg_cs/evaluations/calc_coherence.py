from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.sensing_matrix import (
    Bernoulli,
    BinaryPermutedBlockDiagonal,
    Gaussian,
    SparseBinary,
    Undersampled,
)
from ..models.sparsifying_matrix import CS, DCT, DST, ICS, Gabor, Wavelet


def coherence_test(N: int, M: int, folder: str, random_state: int = 42) -> None:  # noqa: N803
    sensing_matrices = {
        "Gaussian": Gaussian(M, N, random_state),
        "Bernoulli": Bernoulli(M, N, random_state),
        "Undersampled": Undersampled(M, N, random_state),
        "BPBD": BinaryPermutedBlockDiagonal(M, 2),
        "SB": SparseBinary(M, N, d=4, random_state=random_state),
    }

    sparsifying_matrices = {
        "DCT": DCT(N),
        "DST": DST(N),
        "CS": CS(N),
        "ICS": ICS(N),
        "Gabor": Gabor(N, fs=128),
        "Wavelet": Wavelet(N, wavelet="dmey"),
    }

    coherence_table = {}
    for sensing_name, sensing in sensing_matrices.items():
        row = {}
        for sparse_name, sparse in sparsifying_matrices.items():
            coh = sparse.coherence(sensing.value)
            row[sparse_name] = coh
        coherence_table[sensing_name] = row

    df: pd.DataFrame = pd.DataFrame.from_dict(coherence_table, orient="index")
    # df.index.name = "Coerência"
    df.to_csv(path.join(folder, "table.csv"))

    means: list[float] = []
    for sparse_name in sparsifying_matrices:
        vals: list[float] = [
            coherence_table[sensing][sparse_name] for sensing in sensing_matrices
        ]
        means.append(float(np.mean(vals)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        list(sparsifying_matrices.keys()),
        means,
        color=plt.cm.viridis(np.linspace(0.2, 0.8, len(means))),
        edgecolor="black",
    )
    plt.ylabel("Coerência média", fontsize=14)
    plt.xlabel("Dicionário", fontsize=14)
    plt.title("Coerência média para cada dicionário", fontsize=16, pad=15)
    plt.xticks(rotation=25, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 16)
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
        )

    plt.savefig(path.join(folder, "graph.png"), dpi=150)
    plt.show()
    plt.close()


if __name__ == "__main__":
    fs = 128
    signal_length_in_seconds = 4

    N = fs * signal_length_in_seconds
    CR = 2
    M = int(N / CR)

    folder = "eeg_cs/evaluations/coherence"

    coherence_test(N, M, folder=folder, random_state=42)

    df: pd.DataFrame = pd.read_csv(f"{folder}/table.csv", index_col=0)

    df.T.plot(
        kind="bar",
        figsize=(10, 6),
        rot=25,
        ylabel="Coerência média",
        xlabel="Dicionário de Sparsificação",
        title="Coerência média para cada dicionário",
        legend=True,
        fontsize=12,
    )
    plt.savefig(f"{folder}/table.png", dpi=150)
    plt.show()
