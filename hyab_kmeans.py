import os
import numpy as np
import skimage.color
from PIL import Image

MAX_ITER = 10

def srgb_to_lab(srgb):
    return skimage.color.rgb2lab(srgb/255.)

def lab_to_srgb(lab):
    return 255 * skimage.color.lab2rgb(lab)


def quantize_hyab_kmeans(pixels: np.ndarray, num_colors: int, num_iter: int, L_weight = 2.0, L_median=True):
    from sklearn.cluster import kmeans_plusplus

    assert pixels.ndim == 3
    assert pixels.shape[2] == 3
    H, W = pixels.shape[:2]

    X = srgb_to_lab(pixels.reshape(-1, 3))

    N, _ = X.shape
    K = num_colors

    def hyab_distance(a,b):
        L_delta  = a[:, 0]   - b[:, 0]
        ab_delta = a[:, 1:3] - b[:, 1:3]

        L_error  = np.abs(L_delta)
        ab_error = np.sqrt(np.sum(ab_delta**2, axis=1))

        return L_weight * L_error + ab_error


    centers, _ = kmeans_plusplus(X, num_colors, random_state=123)
    indices = np.zeros((H, W), dtype=np.int32)
    dist = np.zeros((N, K))

    for _ in range(num_iter):
        for i in range(K):
            dist[:, i] = hyab_distance(X, centers[i, np.newaxis])

        indices = np.argmin(dist, axis=1)

        if L_median:
            for k in range(K):
                k_inds = indices == k
                if not k_inds.any():
                    continue
                centers[k, 0:1] = np.median(X[k_inds, 0:1], axis=0)
                centers[k, 1:3] = np.mean(X[k_inds, 1:3], axis=0)
        else:
            for k in range(K):
                k_inds = indices == k
                if not k_inds.any():
                    continue
                centers[k] = np.mean(X[k_inds], axis=0)
        

    centers_srgb = lab_to_srgb(centers)
    clusters_8bit = np.clip(np.round(centers_srgb), 0, 255).astype(np.uint8)
    return indices.reshape(H,W), clusters_8bit


def run_hyab_kmeans(img, num_colors:int, num_iter=10, hyab_args={}, dither=0):
    pixels = np.array(img)
    H, W = pixels.shape[:2]

    indices, palette = quantize_hyab_kmeans(pixels, num_colors, num_iter=num_iter, **hyab_args)
    y_hyab = np.take(palette, indices.reshape(-1), axis=0).reshape(H, W, 3)

    return Image.fromarray(y_hyab)


def run_sklearn_kmeans(img: Image.Image, num_colors: int, num_iter: int, verbose=False, in_func=None, out_func=None):
    from sklearn.cluster import KMeans

    array = np.array(img)
    H, W, C = array.shape

    X = array.reshape(-1, C).astype(np.float64)

    if in_func is not None:
        X = in_func(X)

    kmeans = KMeans(
        n_clusters=num_colors,
        init="k-means++",
        n_init=1,
        max_iter=num_iter,
        tol=1e-4,
        verbose=verbose,
        random_state=123,
        algorithm="lloyd",
    )

    indices = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    if out_func is not None:
        centers = out_func(centers)

    palette = np.round(centers).clip(0, 255).astype(np.uint8)

    y = np.take(palette, indices.reshape(-1), axis=0).reshape(H, W, 3)
    return Image.fromarray(y)

def save(img: Image.Image, name: str):
    img.save(os.path.join("output", name + ".png"))

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    img = Image.open("images/caps.png")

    K = 16

    W, H = img.size
    print("Input image  ", img)

    L_weight_big = 2.0

    print("Computing HyAB k-means")
    result_hyab_kmeans1 = run_hyab_kmeans(img, K, num_iter=MAX_ITER, hyab_args={'L_weight': 1.00, 'L_median': False})
    result_hyab_kmeans2 = run_hyab_kmeans(img, K, num_iter=MAX_ITER, hyab_args={'L_weight': L_weight_big, 'L_median': False})
    result_hyab_kmeans3 = run_hyab_kmeans(img, K, num_iter=MAX_ITER, hyab_args={'L_weight': 1.00, 'L_median': True})
    result_hyab_kmeans4 = run_hyab_kmeans(img, K, num_iter=MAX_ITER, hyab_args={'L_weight': L_weight_big, 'L_median': True})

    save(result_hyab_kmeans1, "hyab_kmeans_lw1")
    save(result_hyab_kmeans2, "hyab_kmeans_lw2")
    save(result_hyab_kmeans3, "hyab_kmeans_lw1_med")
    save(result_hyab_kmeans4, "hyab_kmeans_lw2_med")

    print("Computing sRGB k-means")

    # Weighting and internal gamma chosen to match libimagequant
    # See https://github.com/ImageOptim/libimagequant/blob/6aad8f20b28185823813b8bd6823171711480dca/src/pal.rs#L12C1-L19C38
    # Convert from sRGB to internal 1.754 gamma, giving more weight to bright colors.
    # Equal to 0.57/0.4545 = 0.57 / (1/2.2)
    power = 2.2/1.754
    channel_weights = np.array([[0.5, 1.00, 0.45]])

    def srgb_to_weighted_srgb(srgb):
        wsrgb = srgb * channel_weights
        return wsrgb ** power
    
    def weighted_srgb_to_srgb(wsrgb):
        wsrgb = wsrgb ** (1/power)
        wsrgb /= channel_weights
        return wsrgb

    result_srgb_kmeans1 = run_sklearn_kmeans(np.array(img), K, num_iter=MAX_ITER)
    result_srgb_kmeans2 = run_sklearn_kmeans(np.array(img), K, num_iter=MAX_ITER, in_func=srgb_to_weighted_srgb, out_func=weighted_srgb_to_srgb)

    save(result_srgb_kmeans1, "srgb_kmeans")
    save(result_srgb_kmeans2, "srgb_kmeans_weighted")

    print("Computing L*a*b* k-means")

    L_weight_lab = 2.0

    def srgb_to_weighted_lab(srgb):
        lab = srgb_to_lab(srgb)
        lab[...,0] *= L_weight_lab
        return lab

    def weighted_lab_to_srgb(lab):
        lab[...,0] /= L_weight_lab
        srgb = lab_to_srgb(lab)
        return srgb

    result_lab_kmeans1 = run_sklearn_kmeans(img, K, num_iter=MAX_ITER, in_func=srgb_to_lab, out_func=lab_to_srgb)
    result_lab_kmeans2 = run_sklearn_kmeans(img, K, num_iter=MAX_ITER, in_func=srgb_to_weighted_lab, out_func=weighted_lab_to_srgb)

    save(result_lab_kmeans1, "lab_kmeans")
    save(result_lab_kmeans2, "lab_kmeans_weighted")

    # Max coverage example

    result_maxcov_no_dither = img.quantize(K, Image.Quantize.MAXCOVERAGE, kmeans=0, dither=Image.Dither.NONE)
    result_maxcov_dither = img.quantize(K, Image.Quantize.MAXCOVERAGE, kmeans=0, dither=Image.Dither.FLOYDSTEINBERG, palette=result_maxcov_no_dither)

    save(result_maxcov_no_dither, "caps_maxcov_no_dither")
    save(result_maxcov_dither, "caps_maxcov_dither")

    # fig, ax = plt.subplots(1,2,figsize=(16,8))
    # ax[0].axis('off')
    # ax[1].axis('off')
    # ax[0].imshow(result_maxcov_no_dither)
    # ax[1].imshow(result_maxcov_dither)
    # ax[0].set_title("MaxCoverage, no dithering")
    # ax[1].set_title("MaxCoverage, Floyd-Steinberg dithering")
    # fig.suptitle("The effect of dithering")
    # # plt.show()

    fig, ax = plt.subplots(4,2,figsize=(13,16))
    for a in ax.flatten():
        a.axis('off')
    ax[0,0].imshow(img)
    ax[0,0].set_title("sRGB k-means, unweighted").set_color('darkblue')
    ax[1,0].set_title("CIELAB k-means, L_weight=1.0").set_color('brown')
    ax[2,0].set_title("HyAB k-means, L_weight=1.0").set_color('indianred')
    ax[3,0].set_title("HyAB k-means, L_weight=1.0, L_median=True").set_color('darkred')
         
    ax[0,1].set_title("sRGB k-means, weighted").set_color('darkblue')
    ax[1,1].set_title(f"CIELAB k-means, L_weight={L_weight_lab}").set_color('brown')
    ax[2,1].set_title(f"HyAB k-means, L_weight={L_weight_big}").set_color('indianred')
    ax[3,1].set_title(f"HyAB k-means, L_weight={L_weight_big}, L_median=True").set_color('darkred')
         
    ax[0,0].imshow(result_srgb_kmeans1)
    ax[1,0].imshow(result_lab_kmeans1)
    ax[2,0].imshow(result_hyab_kmeans1)
    ax[3,0].imshow(result_hyab_kmeans3)
         
    ax[0,1].imshow(result_srgb_kmeans2)
    ax[1,1].imshow(result_lab_kmeans2)
    ax[2,1].imshow(result_hyab_kmeans2)
    ax[3,1].imshow(result_hyab_kmeans4)

    # ax[4,1].imshow(result_hyab_kmeans)
    plt.suptitle(f"HyAB k-means with a {K=} color palette")
    plt.tight_layout()

    fig.savefig("output/comparison.png")
    plt.show()


    