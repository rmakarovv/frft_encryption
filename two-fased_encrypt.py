import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import dft


def dfrft_matrix(N, alpha):
    p = alpha * np.pi / 2
    W = dft(N)
    eig_vals, eig_vecs = np.linalg.eig(W)
    D = np.diag(np.exp(-1j * p * np.arange(N)))
    return eig_vecs @ D @ np.linalg.inv(eig_vecs)


def generate_random_phase_matrix(shape):
    return np.exp(1j * 2 * np.pi * np.random.rand(*shape))


def dfrft_2d(img, alpha):
    M, N = img.shape
    dfrft_mat_M = dfrft_matrix(M, alpha)
    dfrft_mat_N = dfrft_matrix(N, alpha)

    temp = dfrft_mat_M @ img
    temp = temp @ dfrft_mat_N.conj().T

    return temp


def encrypt_image_2step(img, alpha, beta):
    S = generate_random_phase_matrix(img.shape)
    C = generate_random_phase_matrix(img.shape)

    step1 = img * S
    step2 = dfrft_2d(step1, alpha)
    step3 = step2 * C
    encrypted_img = dfrft_2d(step3, beta)
    return encrypted_img, S, C


def decrypt_image_2step(encrypted_img, S, C, alpha, beta):
    step1 = dfrft_2d(encrypted_img, -beta)
    step2 = step1 * np.conj(C)
    step3 = dfrft_2d(step2, -alpha)
    decrypted_img = step3 * np.conj(S)
    return np.real(decrypted_img)


if __name__ == "__main__":
    img = Image.open("ivan.jpg").convert("L")
    img = np.array(img, dtype=float) / 255.0

    alpha = 0.6
    beta = 1.2

    encrypted_img, S, C = encrypt_image_2step(img, alpha, beta)
    decrypted_img = decrypt_image_2step(encrypted_img, S, C, alpha, beta)

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
