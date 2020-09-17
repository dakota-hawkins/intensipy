# %%
import typing

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, signal, stats
from skimage import exposure, filters, morphology
from statsmodels.distributions.empirical_distribution import ECDF


class Intensify:

    def __init__(self, t=None, dy=None, dx=None, n_quantiles=10000,
                 smooth_quartiles=True):
        self.t = t
        self.dy = dy
        self.dx = dx
        self.n_quantiles = n_quantiles
        self.smooth_quartiles = smooth_quartiles


# ----------------------------- Properties -------------------------------------
    @property
    def t(self):
        return self.t_

    @t.setter
    def t(self, value):
        if not Intensify.check_numeric(value) and value is not None:
            raise ValueError(f"`t` should be numeric. Received {type(value)}.")
        self.t_ = value

    @property
    def dy(self):
        return self.dy_

    @dy.setter
    def dy(self, value):
        if not Intensify.check_numeric(value) and value is not None:
            raise ValueError(f"`dy` should be numeric. Recieved {type(value)}.")
        if value is None:
            value = 7
        self.dy_ = value

    @property
    def dx(self):
        return self.dx_
    
    @dx.setter
    def dx(self, value):
        if not Intensify.check_numeric(value) and value is not None:
            raise ValueError(f"`dx` should be numeric. Recieved {type(value)}.")
        if value is None:
            value = 7
        self.dx_ = value

    @property
    def n_quantiles(self):
        return self.n_quantiles_
    
    @n_quantiles.setter
    def n_quantiles(self, value):
        if not isinstance(value, (int, np.integer)):
            raise ValueError("Expected integer values for `n_quantile`."\
                             f" Received {type(value)}.")
        self.n_quantiles_ = value
        
    @property
    def smooth_quartiles(self):
        return self.smooth_quartiles_
    
    @smooth_quartiles.setter
    def smooth_quartiles(self, value):
        if not isinstance(value, bool):
            raise ValueError("Expected boolean value for `smooth_quartiles`."\
                             f" Received {type(value)}.")
        self.smooth_quartiles_ = value 
        

# ------------------------------ Class Functions -------------------------------
    def normalize(self, z_stack, ref_idx=None, smooth=True, verbose=True):                
        # set defaults if no user specified parameters
        if smooth:
            z_stack = np.array([filters.gaussian(x) for x in z_stack])
        if self.t is not None:
            cutoff = stats.percentileofscore(np.array(dapi[ref_idx]).flatten(),
                                             self.t)
            thresholds = np.array([np.percentile(x.flatten(), cutoff)\
                               for x in z_stack])
        else:
            thresholds = np.array([filters.threshold_otsu(np.array(x))\
                                   for x in z_stack])

        # create empty z-stack for normalized data
        normed = np.zeros_like(z_stack, dtype='float')
        # create variables to track quantile values
        semi_quantiles = np.zeros((z_stack.shape[0], self.n_quantiles))
        p_quantiles = np.arange(0, 100, 100 / self.n_quantiles)
        # normalize along xy axis for each image.
        if verbose:
            print("Normalizing intensity values across xy-dimensions.")
        for i in Intensify.get_iterator(z_stack.shape[0], verbose):
            normed[i], thresholds[i] = self.xy_normalize(np.array(z_stack[i]),
                                                         thresholds[i])
            semi_quantiles[i, :] = np.percentile(normed[i][normed[i] < thresholds[i]],
                                                 p_quantiles)
        # create variables for semi-quantile normalization
        agg_quantiles = np.percentile(semi_quantiles, 98, axis=0)
        semi_median = agg_quantiles[self.n_quantiles // 2]
        semi_max = agg_quantiles[-1]
        transform = interpolate.interp1d(np.arange(self.n_quantiles),
                                         agg_quantiles)
        if verbose:
            print("Normalizing intensity values across z-dimension.")
        for i in Intensify.get_iterator(z_stack.shape[0], verbose):
            normed[i] = self.z_normalize(normed[i], thresholds[i], transform,
                                         semi_median, semi_max)
        return normed

    def xy_normalize(self, img, t, smooth='savitzky-galore'):
        # create mask image for xy normalization

        fg = img >= t
        # replace high signal region with randomly sampled background
        # intensities
        mask = self.smooth_background(img, fg, smooth)
        mask /= mask.max()
        # divide normalize original image
        norm_xy = img / mask
        # standardize to old values
        norm_xy /= (np.median(norm_xy) / np.median(img))
        new_t = norm_xy[np.logical_and(fg, norm_xy > 0)].min()
        return (norm_xy, new_t)

    def smooth_background(self, img, selected, method):
        mask = img.copy()
        mask[selected] = Intensify.sample_background(img[~selected],
                                                     selected.sum())
        if method == 'savitzky-galore':
            mask = Intensify.savitzky_galoy(mask, dy=self.dy, dx=self.dx, k=1)
        elif method == 'gaussian':
            mask = filters.gaussian(mask, 2)
        elif method == 'median':
            mask = filters.median(mask, selem=morphology.disk(min(self.dx, self.dy)))
        else:
            raise ValueError(f"Unsupported smoothing operation: {method}.")
        return mask

    def z_normalize(self, img, t, transform, semi_median, semi_max):
        (lower, upper) = np.percentile(img[img > 0], [25, 99])
        fg = img >= t
        # perform constrast stretching on foreground pixels
        img[fg] = Intensify.constrast_stretch(img[fg], lower, upper,
                                              semi_median, semi_max)
        # quantile normalization on background pixels
        img[~fg] = Intensify.quantile_normalization(img[~fg],
                                                    self.n_quantiles,
                                                    transform)
        # re-scale foreground intensities
        img[fg] *= np.max(img[~fg]) / np.min(img[fg])
        # quantile normalization produces a decent amount of static:
        # suggested to smooth similar to xy smoothing of background
        if self.smooth_quartiles:
            mask = self.smooth_background(img, fg, 'savitzky-galore')
            img[~fg] = mask[~fg]
        return img


# ----------------------------- Static Helper Functions ------------------------
    @staticmethod
    def check_numeric(value):
        try:
            float(value)
        except :
            return False
        return True

    @staticmethod
    def get_iterator(n, verbose):
        zrange = range(n)
        if verbose:
            try:
                from tqdm import trange
                zrange = trange(n)
            except ImportError:
                pass
        return zrange

    @staticmethod
    def sample_background(intensities, n):
        p1 = np.percentile(intensities, 10)
        p2 = np.percentile(intensities, 90)
        samples = intensities[np.logical_and(p1 < intensities,
                                             intensities < p2)]
        ecdf = ECDF(samples)
        return ecdf.x[np.searchsorted(ecdf.y, np.random.uniform(size=n))] 

    @staticmethod
    def quantile_normalization(pixels, n_quantiles, transform):
        # transform pixel ranks to aggregate quantile space
        values = transform(np.linspace(0, n_quantiles - 1, pixels.size))
        return values[np.argsort(pixels)]

    @staticmethod
    def constrast_stretch(pixels, lower, upper, semi_median, semi_max):
        """
        Constrast stretch pixels following methods outline in original paper.
        
        Parameters:
            pixels: np.ndarray
            lower: float
                Intensity value corresponding to the 1st quartile of `pixels`.
            upper: float
                Intensity value corresponding to the 98th percentile of `pixels`.
            semi_median: float
                Intensity value corresponding to the median of semi-quantiles.
            semi_max: float
                Intensity value corresponding to the maximum of semi-quantiles.
        Return
            np.ndarray
                Adjusted pixels
        """
        out = (pixels - lower)\
            * ((semi_max - semi_median) / (upper - lower))\
            + semi_median
        return out

    @staticmethod
    def savitzky_galoy(img, dy, dx, k):
        by_row = signal.savgol_filter(img, dy, k, deriv=0,
                                      delta=1.0, axis=0, mode='interp',
                                      cval=0.0)
        by_col = signal.savgol_filter(by_row, dx, k, deriv=0,
                                      delta=1.0, axis=1, mode='interp',
                                      cval=0.0)
        return by_col


# %%
if __name__ == '__main__':
    model = Intensify()
    from dask_image.imread import imread
    import dask.array as da
    import napari

    data_dir = "/home/dakota/Pictures/Embryos/Otop/18hpf_otop2L_emb1.oif.files/"
    dapi = imread(data_dir + 's_C001*.tif')[2:-2, : , :]
    z_stack = np.array(dapi)
    model = Intensify(dx=29, dy=29, smooth_quartiles=True)
    normed = model.normalize(z_stack)
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(dapi, #contrast_limits=[0,2000],
                         scale=[4.7, 1, 1], name="dapi")
        viewer.add_image(da.array(normed), scale=[4.7, 1, 1],
                         name='normed')
# %%
