# %%
import matplotlib.pyplot as plt
import numpy as np
from models.compressed_sensing import CompressedSensing
from models.loader import CHBMITLoader
from models.reconstruction_algorithm import (
    SimultaneousOrthogonalMatchingPursuit,
    SPGL1BasisPursuitDenoising,
)
from models.sensing_matrix import (
    Bernoulli,
    Gaussian,
)
from models.sparsifying_matrix import DCT, Wavelet


def main():
    # ----------------------------------1° Etapa: Aquisição dos dados ----------------------
    random_state = 42
    segment_length_sec = 4
    n_blocks = 500

    loader = CHBMITLoader(n_blocks, segment_length_sec, random_state)
    # loader = BCIIVLoader(n_blocks, segment_length_sec, random_state)
    # loader = BCIIIILoader(n_blocks, segment_length_sec, random_state)

    # fs = 256
    # fs = 240
    fs = 128
    loader.downsample(new_fs=fs)

    data = loader.data  # (blocks, samples, channels)

    N = data.shape[1]  # Number of samples in each block
    CR = 2  # Compressive Ratio
    M = int(N / CR)

    # ----------------------------------2° Etapa: Definir a matriz de amostragem ----------------------
    # sensing_matrix = SparseBinary(M, N, d=8, random_state=random_state)
    sensing_matrix = Bernoulli(M, N, random_state=random_state)
    # sensing_matrix = Gaussian(M, N, random_state=random_state)
    # sensing_matrix = Undersampled(M, N, random_state=random_state)
    # sensing_matrix = BinaryPermutedBlockDiagonal(M, CR)

    # ----------------------------------3° Etapa: Definir dicionário ----------------------------------
    sparse_bases = DCT(N)
    # sparse_bases = DST(N)
    # sparse_bases = CS(N)
    # sparse_bases = ICS(N)
    # sparse_bases = Gabor(N, fs=fs, tf=2, ff=4)
    # sparse_bases = Wavelet(N, wavelet='dmey')

    # sparse_bases = KSVDBasis.load("./dictionaries/chbmit/1749116724410703057_ksvd_3000.pkl")

    # training_data, test_data = loader.split_training_and_test(n_blocks=350, flatten=True, remove_mean=True)
    # data = test_data
    # segments = loader.get_random_segments(n_segments=8000, random_state=random_state)

    # sparse_bases = KSVDBasis(training_data=segments, n_atoms=4000, n_coefficients=30, max_iter=80)
    # sparse_bases.save(f"./dictionaries/{loader.dataset}")
    # ----------------------------------4° Etapa: Definir algoritmo de reconstrução -------------------
    # reconstruction_algorithm = BasisPursuit(solver='ECOS')
    # reconstruction_algorithm = BasisPursuitDenoising()
    reconstruction_algorithm = SPGL1BasisPursuitDenoising(
        sigma_factor=0.0001, max_iter=10000
    )
    # reconstruction_algorithm = OrthogonalMatchingPursuit(n_nonzero_coefs=250)
    # reconstruction_algorithm = SPGL1BasisPursuit(max_iter=10000)
    # reconstruction_algorithm = SimultaneousOrthogonalMatchingPursuit(max_iter=1000, tol=1e-8)

    # ----------------------------------5° Etapa: Aplicar compressed sensing  -------------------------
    cs = CompressedSensing(sensing_matrix, sparse_bases, reconstruction_algorithm)

    for X in data:  # Process only the first 10 blocks for demonstration
        Y, Theta = cs.compress(X)
        X_hat = cs.reconstruct(Y, Theta)

        # ----------------------------------6° Etapa: Avaliar resultado -----------------------------------
        coherence = cs.sparse_bases.coherence(cs.sensing_matrix.value)
        print(f"coherence = {coherence}")

        prd, nmse, sndr = CompressedSensing.evaluate(X, X_hat)
        print(f"prd = {prd}", f"nmse = {nmse}", f"sndr = {sndr}")
        print(
            f"prd mean = {prd.mean()}",
            f"nsme mean = {nmse.mean()}",
            f"sndr mean = {sndr.mean()}",
        )
        print(f"error = {np.linalg.norm(X - X_hat, axis=0).mean()}")

        # ----------------------------------7° Etapa: Plotar curvas ---------------------------------------
        # fig, ax = plt.subplots(figsize=[15, 5])
        # ax.plot(X[:, 3], 'k', label='Sinal original')
        # ax.plot(X_hat[:, 0], 'r', linestyle='dotted', label='Sinal reconstruído')
        # ax.set_xlabel('Tempo (s)')
        # ax.set_ylabel('Amplitude')
        # ax.set_title('Sinal original x Sinal reconstruído')
        # ax.legend(loc='upper right', framealpha=1)
        # ax.grid()

        # plt.show()

        # fig, ax = plt.subplots(figsize=[15, 5])
        # # ax.plot(t, X[:, 0], 'k', label='Sinal original')
        # ax.plot(Y[:, 3], 'r', linestyle='dotted', label='Sinal amostrado')
        # ax.set_xlabel('Tempo (s)')
        # ax.set_ylabel('Amplitude')
        # ax.set_title('Sinal original x Sinal amostrado')
        # ax.legend(loc='upper right', framealpha=1)
        # ax.grid()

        # plt.show()


# if __name__ == "__main__":
#     main()


# ----------------------------------1° Etapa: Aquisição dos dados ----------------------
random_state = 80
segment_length_sec = 4
n_blocks = 2

loader = CHBMITLoader(n_blocks, segment_length_sec, random_state)
# loader = BCIIVLoader(n_blocks, segment_length_sec, random_state)
# loader = BCIIIILoader(n_blocks, segment_length_sec, random_state)

# fs = 256
# fs = 240
fs = 128
loader.downsample(new_fs=fs)

data = loader.data  # (blocks, samples, channels)

N = data.shape[1]  # Number of samples in each block
CR = 2  # Compressive Ratio
M = int(N / CR)

# %%
# ----------------------------------2° Etapa: Definir a matriz de amostragem ----------------------
# sensing_matrix = SparseBinary(M, N, d=16, random_state=random_state)
# sensing_matrix = Bernoulli(M, N, random_state=random_state)
sensing_matrix = Gaussian(M, N, random_state=random_state)
# sensing_matrix = Undersampled(M, N, random_state=random_state)
# sensing_matrix = BinaryPermutedBlockDiagonal(M, CR)

# ----------------------------------3° Etapa: Definir dicionário ----------------------------------
# sparse_bases = DCT(N)
# sparse_bases = DST(N)
# sparse_bases = CS(N)
# sparse_bases = ICS(N)
# sparse_bases = Gabor(N, fs=fs, tf=2, ff=4)
sparse_bases = Wavelet(N, wavelet="dmey")

# sparse_bases = KSVDBasis.load("./dictionaries/chbmit/1749116724410703057_ksvd_3000.pkl")

# training_data, test_data = loader.split_training_and_test(n_blocks=350, flatten=True, remove_mean=True)
# data = test_data
# segments = loader.get_random_segments(n_segments=8000, random_state=random_state)

# sparse_bases = KSVDBasis(training_data=segments, n_atoms=4000, n_coefficients=30, max_iter=80)
# sparse_bases.save(f"./dictionaries/{loader.dataset}")
# ----------------------------------4° Etapa: Definir algoritmo de reconstrução -------------------
# reconstruction_algorithm = BasisPursuit(solver='ECOS')
# reconstruction_algorithm = BasisPursuitDenoising()
# reconstruction_algorithm = SPGL1BasisPursuitDenoising(sigma_factor=0.0001, max_iter=10000)
# reconstruction_algorithm = OrthogonalMatchingPursuit(n_nonzero_coefs=250)
# reconstruction_algorithm = SPGL1BasisPursuit(max_iter=10000)
reconstruction_algorithm = SimultaneousOrthogonalMatchingPursuit(
    max_iter=1000, tol=1e-8
)

# ----------------------------------5° Etapa: Aplicar compressed sensing  -------------------------
cs = CompressedSensing(sensing_matrix, sparse_bases, reconstruction_algorithm)

for X in data:  # Process only the first 10 blocks for demonstration
    Y, Theta = cs.compress(X)
    X_hat = cs.reconstruct(Y, Theta)

    # ----------------------------------6° Etapa: Avaliar resultado -----------------------------------
    coherence = cs.sparse_bases.coherence(cs.sensing_matrix.value)
    print(f"coherence = {coherence}")

    prd, nmse, sndr = CompressedSensing.evaluate(X, X_hat)
    print(f"prd = {prd}", f"nmse = {nmse}", f"sndr = {sndr}")
    print(
        f"prd mean = {prd.mean()}",
        f"nsme mean = {nmse.mean()}",
        f"sndr mean = {sndr.mean()}",
    )
    print(f"error = {np.linalg.norm(X - X_hat, axis=0).mean()}")

    # ----------------------------------7° Etapa: Plotar curvas ---------------------------------------
    # fig, ax = plt.subplots(figsize=[15, 5])
    # ax.plot(X[:, 3], 'k', label='Sinal original')
    # ax.plot(X_hat[:, 0], 'r', linestyle='dotted', label='Sinal reconstruído')
    # ax.set_xlabel('Tempo (s)')
    # ax.set_ylabel('Amplitude')
    # ax.set_title('Sinal original x Sinal reconstruído')
    # ax.legend(loc='upper right', framealpha=1)
    # ax.grid()

    # plt.show()

    # fig, ax = plt.subplots(figsize=[15, 5])
    # # ax.plot(t, X[:, 0], 'k', label='Sinal original')
    # ax.plot(Y[:, 3], 'r', linestyle='dotted', label='Sinal amostrado')
    # ax.set_xlabel('Tempo (s)')
    # ax.set_ylabel('Amplitude')
    # ax.set_title('Sinal original x Sinal amostrado')
    # ax.legend(loc='upper right', framealpha=1)
    # ax.grid()

    # plt.show()
    break

# %%
for i in range(X.shape[1]):
    print(f"prd = {prd[i]}")
    fig, ax = plt.subplots(figsize=[15, 5])
    ax.plot(X[:, i], "k", label="Sinal original")
    ax.plot(X_hat[:, i], "r", linestyle="dotted", label="Sinal reconstruído")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sinal original x Sinal reconstruído")
    ax.legend(loc="upper right", framealpha=1)
    ax.grid()

    plt.show()

    fig, ax = plt.subplots(figsize=[15, 5])
    ax.plot(Y[:, i], "r", linestyle="dotted", label="Sinal amostrado")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sinal original x Sinal amostrado")
    ax.legend(loc="upper right", framealpha=1)
    ax.grid()

    plt.show()
# %%
