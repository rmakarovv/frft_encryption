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


def encrypt_rand(f, g, Dx, Dy):
    g_scrambled, perm = pixel_scrambling(g)
    g_phase = phase_encoding(g_scrambled)
    C = f * g_phase
    psi = drft_image(C, Dx, Dy)
    return psi, perm


def decrypt_rand(psi, Dx, Dy, perm):
    C = inverse_drft_image(psi, Dx, Dy)
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

    Dx = random_diagonal_matrix(f.shape[0])
    Dy = random_diagonal_matrix(f.shape[1])

    encrypted_image_r, perm_r = encrypt_rand(f, g, Dx, Dy)
    f_decrypted_r, g_decrypted_r = decrypt_rand(encrypted_image_r, Dx, Dy, perm_r)

    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    ax[0, 0].imshow(f, cmap="gray")
    ax[0, 0].set_title("Original image")
    ax[0, 0].set_axis_off()
    ax[1, 0].imshow(g, cmap="gray")
    ax[1, 0].set_title("Original image, phase")
    ax[1, 0].set_axis_off()
    ax[0, 1].imshow(np.abs(encrypted_image_r), cmap="gray")
    ax[0, 1].set_title("Encrypted image")
    ax[0, 1].set_axis_off()
    ax[0, 2].imshow(f_decrypted_r, cmap="gray")
    ax[0, 2].set_title("Decrypted image")
    ax[0, 2].set_axis_off()
    ax[1, 2].imshow(g_decrypted_r, cmap="gray")
    ax[1, 2].set_title("Decrypted image, phase")
    ax[1, 2].set_axis_off()
    ax[1, 1].set_axis_off()
    plt.show()
