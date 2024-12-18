import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def compute_dft_eigenvectors(N):
    k, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2j * np.pi / N)
    V = np.power(omega, k * n) / np.sqrt(N)
    return V


def random_diagonal_matrix(N):
    random_phases = np.exp(1j * 2 * np.pi * np.random.rand(N))
    D_R = np.diag(random_phases)
    return D_R


def drft(signal):
    N = len(signal)
    V = compute_dft_eigenvectors(N)
    D_R = random_diagonal_matrix(N)
    transformed_signal = V @ D_R @ V.T.conj() @ signal
    return transformed_signal


def drft_image(image, D_x, D_y):
    N, M = image.shape
    V_x = compute_dft_eigenvectors(N)
    V_y = compute_dft_eigenvectors(M)
    transformed_image = V_x @ D_x @ V_x.T.conj() @ image @ V_y @ D_y @ V_y.T.conj()
    return transformed_image


def inverse_drft(transformed_signal, random_matrix):
    N = transformed_signal.size
    V = compute_dft_eigenvectors(N)
    D_R_inverse = np.conj(random_matrix)
    reconstructed_signal = V @ D_R_inverse @ V.T.conj() @ transformed_signal
    return reconstructed_signal


def inverse_drft_image(transformed_image, random_matrix_x, random_matrix_y):
    N, M = transformed_image.shape
    V_x = compute_dft_eigenvectors(N)
    V_y = compute_dft_eigenvectors(M)
    D_x_inverse = np.conj(random_matrix_x)
    D_y_inverse = np.conj(random_matrix_y)
    reconstructed_image = (
        V_x
        @ D_x_inverse
        @ V_x.T.conj()
        @ transformed_image
        @ V_y
        @ D_y_inverse
        @ V_y.T.conj()
    )
    return reconstructed_image


if __name__ == "__main__":
    img = Image.open("ivan.jpg").convert("L")
    img = np.array(img, dtype=float) / 255.0

    alpha = 0.6
    beta = 1.2

    random_matrix_x = random_diagonal_matrix(img.shape[0])
    random_matrix_y = random_diagonal_matrix(img.shape[1])

    encrypted_img = drft_image(img, random_matrix_x, random_matrix_y)
    decrypted_img = inverse_drft_image(encrypted_img, random_matrix_x, random_matrix_y)
    decrypted_img = np.abs(decrypted_img)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray", aspect="auto")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Encrypted Image")
    plt.imshow(np.abs(encrypted_img), cmap="gray", aspect="auto")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Decrypted Image")
    plt.imshow(decrypted_img, cmap="gray", aspect="auto")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
