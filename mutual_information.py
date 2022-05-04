import numpy as np


def calculate_mutual_information(im1, im2):
    N = 255
    im1 = (im1 + 1) * N / 2
    im1 = np.around(im1)
    im1 = np.clip(im1, 0, N)

    im2 = (im2 + 1) * N / 2
    im2 = np.around(im2)
    im2 = np.clip(im2, 0, N)

    h1 = np.histogram(im1.ravel(), N+1, [0, N + 1], density=True)[0]
    h2 = np.histogram(im2.ravel(), N+1, [0, N + 1], density=True)[0]
    joint = np.histogram2d(im1.ravel(), im2.ravel(), bins=(N+1, N+1),range=[[0, N + 1],[0, N + 1]], density=True)[0]
    # print(joint.shape)
    sum = 0
    for x in range(N+1):
        for y in range(N+1):
            p_xy = joint[x, y]
            p_x = h1[x]
            p_y = h2[y]
            # print(x,y, p_x,p_y,p_x*p_y,p_xy)
            sum += p_xy * np.log2((p_xy + 1e-10) / (p_x * p_y + 1e-10))
            # if np.isnan(sum):
            #    return
    return sum


im1 = np.random.rand(400, 400)
im2 = np.random.rand(400, 400)
print(calculate_mutual_information(im1, im2))
