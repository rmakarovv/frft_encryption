import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import dft
from skimage.transform import rescale


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


def pixel_scrambling(image):
    flat_image = image.flatten()
    perm = np.random.permutation(len(flat_image))
    return flat_image[perm].reshape(image.shape), perm


def inverse_pixel_scrambling(scrambled_image, perm):
    flat_scrambled_image = scrambled_image.flatten()
    inverse_perm = np.argsort(perm)
    original_flat_image = flat_scrambled_image[inverse_perm]
    return original_flat_image.reshape(scrambled_image.shape)


def phase_encoding(image):
    return np.exp(1j * np.pi * image)


def random_phase_mask(size):
    return np.exp(1j * 2 * np.pi * np.random.rand(*size))


def encrypt(f, g, alpha, beta, mask1, mask2):
    g_scrambled, perm = pixel_scrambling(g)
    g_phase = phase_encoding(g_scrambled)
    C = f * g_phase
    C_masked = C * mask1
    C_frft = dfrft_2d(C_masked, alpha)
    C_frft_masked = C_frft * mask2
    psi = dfrft_2d(C_frft_masked, beta - alpha)
    return psi, perm


def decrypt(psi, alpha, beta, mask1, mask2, perm):
    C_frft_masked = dfrft_2d(psi, alpha - beta)
    C_frft = C_frft_masked * np.conj(mask2)
    C_masked = dfrft_2d(C_frft, -alpha)
    C = C_masked * np.conj(mask1)
    f_decrypted = np.abs(C)
    g_phase = np.angle(C) / np.pi
    g_decrypted = inverse_pixel_scrambling(g_phase, perm)
    return f_decrypted, g_decrypted


if __name__ == "__main__":
    f = cv2.imread("ivan.jpg", cv2.IMREAD_GRAYSCALE) / 255.0
    g = cv2.imread("message.png", cv2.IMREAD_GRAYSCALE) / 255.0

    f = rescale(f, 0.3, anti_aliasing=True)
    g = rescale(g, 0.3, anti_aliasing=True)

    alpha = 0.8
    beta = 0.4
    mask1 = random_phase_mask(f.shape)
    mask2 = random_phase_mask(f.shape)

    encrypted_image, perm = encrypt(f, g, alpha, beta, mask1, mask2)

    f_decrypted, g_decrypted = decrypt(encrypted_image, alpha, beta, mask1, mask2, perm)

    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].imshow(f, cmap="gray")
    ax[0, 0].set_title("Original image")
    ax[0, 0].set_axis_off()
    ax[1, 0].imshow(g, cmap="gray")
    ax[1, 0].set_title("Original image, phase")
    ax[1, 0].set_axis_off()
    ax[0, 1].imshow(np.abs(encrypted_image), cmap="gray")
    ax[0, 1].set_title("Encrypted image")
    ax[0, 1].set_axis_off()
    ax[0, 2].imshow(f_decrypted, cmap="gray")
    ax[0, 2].set_title("Decrypted image")
    ax[0, 2].set_axis_off()
    ax[1, 2].imshow(g_decrypted, cmap="gray")
    ax[1, 2].set_title("Decrypted image, phase")
    ax[1, 2].set_axis_off()
    ax[1, 1].set_axis_off()
    plt.show()
