import cv2 as cv
import numpy as np
import math

path = #insert image path
src = cv.imread(path, cv.IMREAD_GRAYSCALE)

kernel1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel2 = np.ones((5, 5)) / 25

sobel = cv.filter2D(src, -1, kernel1)
preprocessed_image = cv.filter2D(sobel, -1, kernel2)

[N, M] = np.shape(src)


def IVAR(matrix):
    k = 2
    flatten_matrix = np.ravel(matrix)
    sorted_matrix = np.sort(flatten_matrix, axis=None)[::-1]
    G_i = []
    for i in range(0, k):
        G_i.append(sorted_matrix[i])
    m_i = np.mean(flatten_matrix)
    ivar = []
    for i in range(0, k):
        ivar.append(math.pow(G_i[i] - m_i, 2))
    return np.round(np.sum(ivar))


def IMEAN(matrix):
    k = 2
    flatten_matrix = np.ravel(matrix)
    sorted_matrix = np.sort(flatten_matrix, axis=None)[::-1]
    G_i = []
    for i in range(0, k):
        G_i.append(sorted_matrix[i])
    imean = np.sum(G_i) / k
    return imean


W = 10
m = int(np.floor((2 * (M - W) / W) + 1))
n = int(np.floor((2 * (N - W) / W) + 1))

NLCM = np.zeros((n, m), dtype=np.float32)
IMEANi = np.zeros((n, m), dtype=np.float32)
IVARi = np.zeros((n, m), dtype=np.float32)

row = 0
column = 0
for i in range(0, M - W + 1, int(np.floor(W / 2))):
    column = 0
    for j in range(0, N - W + 1, int(np.floor(W / 2))):
        sliding_window = preprocessed_image[j:j + W, i:i + W]
        IVARi[column, row] = IVAR(np.copy(sliding_window))
        IMEANi[column, row] = IMEAN(np.copy(sliding_window))
        column += 1
    row += 1

for i in range(0, m):
    for j in range(0, n):
        IVAR_sliding_window = IVARi[j:j + 3, i:i + 3]
        IMEAN_sliding_window = IMEANi[j:j + 3, i:i + 3]
        if np.shape(IVAR_sliding_window)[0] == 2 or np.shape(IVAR_sliding_window)[0] == 1 or \
                np.shape(IVAR_sliding_window)[1] == 2 or np.shape(IVAR_sliding_window)[1] == 1:
            continue
        IVARu = np.ravel(IVAR_sliding_window)[4]
        IMEAN_vector = list(np.ravel(IVAR_sliding_window))
        IMEANu = IMEAN_vector.pop(4)

        nlcm = []
        for imeani in IMEAN_vector:
            if imeani == 0:
                nlcm.append(0)
            else:
                nlcm.append(abs((IVARu * IMEANu) / imeani))
        NLCM[j, i] = np.min(nlcm)

K = 12
std = np.floor(np.std(NLCM))
mean = np.floor(np.mean(NLCM))
T = mean + (K * std)

NLCM = NLCM < T

X = []
Y = []
for i in range(0, m):
    for j in range(0, n):
        if not NLCM[j, i]:
            for x in range(i - 3, i + 4):
                for y in range(j - 3, j + 4):
                    if not NLCM[y, x]:
                        X.append(int(W * x / 2) + W)
                        Y.append(int(W * y / 2) + W)
            coordinate_x_min = np.min(X)
            coordinate_y_min = np.min(Y)
            coordinate_x_max = np.max(X)
            coordinate_y_max = np.max(Y)
            final_img = cv.rectangle(src, (coordinate_x_min, coordinate_y_min),
                                     (coordinate_x_max + W, coordinate_y_max + W), (0, 0, 255), 1)
            X = []
            Y = []

cv.imwrite('img.png', final_img)
cv.imshow('final', final_img)
k = cv.waitKey(0)
