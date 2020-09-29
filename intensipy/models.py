"""
Normalize intensity in 3D image stacks.

Implements the Intensify3D algorithm as described by Yoyan et al. 

References
----------
1.Yayon, N. et al. Intensify3D: Normalizing signal intensity in large
heterogenic image stacks. Scientific Reports 8, 4311 (2018).

@author Dakota Y. Hawkins
"""
# %%
# TODO: put type hinting
import typing 

import numpy as np
from scipy import interpolate, signal, stats
from skimage import exposure, filters, morphology
from statsmodels.distributions.empirical_distribution import ECDF


class Intensify:
    """Class to perform Intensify3D over a 3-Dimensional image stack."""

    def __init__(self, xy_norm=True, z_norm=True,
                 t=None, dy=None, dx=None, n_quantiles=10000,
                 smooth_quartiles=True, keep_original_scale=True,
                 stretch_method='skimage',
                 bits=12):
        """
        Normalizes 3D image stacks using Intensify3D.

        Parameters
        ----------
        xy_norm : bool, optional
            Whether to normalize image slices along the XY plane. Default True.
        z_norm : bool, optional
            Whether to normalize image slices along the Z plane. Default True.
        dy : int, optional
            Y-diameter of expected object shape. Used for smoothing.
            Default is 29.
        dx : int, optional
            X-diameter of expected object shape. Used for smoothing.
            Default is 29.
        n_quantiles : int, optional
            Number of quantiles to calculate during semi-quantile normalization.
            By default 10,000 per original paper.
        smooth_quartiles : bool, optional
            Whether to smooth quantile normalized pixels, by default True. To
            recapitulate original paper set to False.
        keep_original_scale : bool, optional
            Whether minimum and maximum values should be consistent with the
            original image bit depth. Default is True.
        stretch_method : str, optional
            Method used to perform contrast stretching. Default is 'skimage'. To
            run origianl Intensipy3D implementation -- as best discerned via
            source code -- set to 'intensify3d'.
        bits : int, optional
            Bit depth of original image. Default is 12.

        References
        ----------
        1.Yayon, N. et al. Intensify3D: Normalizing signal intensity in large
        heterogenic image stacks. Scientific Reports 8, 4311 (2018).
        """
        self.xy_norm = xy_norm
        self.z_norm = z_norm
        self.dy = dy
        self.dx = dx
        self.n_quantiles = n_quantiles
        self.smooth_quartiles = smooth_quartiles
        self.keep_original_scale = keep_original_scale
        self.stretch_method = stretch_method
        self.bits = bits

# ----------------------------- Properties -------------------------------------
    @property
    def xy_norm(self):
        """Whether to normalize image slices along the XY plane."""
        return self.xy_norm_

    @xy_norm.setter
    def xy_norm(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected boolean value for `xy_norm`.")
        self.xy_norm_ = value

    @property
    def z_norm(self):
        """Whether to normalize image slices along the Z plane."""
        return self.z_norm_

    @z_norm.setter
    def z_norm(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected boolean value for `z_norm`.")
        self.z_norm_ = value
    
    @property
    def dy(self):
        """Y-radius of expected object shape. Used for smoothing."""
        return self.dy_

    @dy.setter
    def dy(self, value):
        if not Intensify.check_numeric(value) and value is not None:
            raise TypeError(f"`dy` should be numeric. Recieved {type(value)}.")
        if value is None:
            value = 29
        self.dy_ = value

    @property
    def dx(self):
        """X-radius of expected object shape. Used for smoothing."""
        return self.dx_
    
    @dx.setter
    def dx(self, value):
        if not Intensify.check_numeric(value) and value is not None:
            raise TypeError(f"`dx` should be numeric. Recieved {type(value)}.")
        if value is None:
            value = 29
        self.dx_ = value

    @property
    def n_quantiles(self):
        """Number of quantiles to calculate during semi-quantile normalization."""
        return self.n_quantiles_
    
    @n_quantiles.setter
    def n_quantiles(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("Expected integer values for `n_quantile`."\
                             f" Received {type(value)}.")
        self.n_quantiles_ = value
        
    @property
    def smooth_quartiles(self):
        """Whether to smooth quantile normalized pixels."""
        return self.smooth_quartiles_
    
    @smooth_quartiles.setter
    def smooth_quartiles(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected boolean value for `smooth_quartiles`."\
                             f" Received {type(value)}.")
        self.smooth_quartiles_ = value 

    @property
    def keep_original_scale(self):
        """Whether to return images in original intensity scale."""
        return self.keep_original_scale_

    @keep_original_scale.setter
    def keep_original_scale(self, value):
        if not isinstance(value, bool):
            raise TypeError("Expected boolean value for `keep_original_scale`."\
                             f" Received {type(value)}.")
        self.keep_original_scale_ = value

    @property
    def stretch_method(self):
        """Method to perform constrast stretching."""
        return self.stretch_method_

    @stretch_method.setter
    def stretch_method(self, value):
        if not isinstance(value, str):
            raise TypeError("Expected string for `stretch_method`.")
        if value.lower() not in ['skimage', 'intensify3d']:
            raise ValueError(f"{value} not a valid constrast-stretch method."\
                              "Supported methods are 'skimage' or 'intensify3d.")
        self.stretch_method_ = value.lower()

    @property
    def bits(self):
        """Image bit depth. Only necessary if `keep_original_scale=True`."""
        return self.bits_
    
    @bits.setter
    def bits(self, value):
        if float(value) != int(value):
            raise TypeError("Expected integer value for `bits`.")
        self.bits_ = int(value)
        
# ------------------------------ Class Functions -------------------------------
    def normalize(self, z_stack, t=None, ref_idx=None, verbose=True):
        """
        Normalize a 3-Dimensional image stack along xy + z dimensions.

        Parameters
        ----------
        z_stack : array, optional
            Numerical array of 3D image. Expects ZXY order for axes.
        t : int, optional
            Maximum background intensity (MBI) for reference image. Requires
            `ref_idx` to be set. If `t` is not provided, MBIs are estimated for
            each image using the otsu method.
        ref_idx : int, optional
            Which image slice in `z_stack`. Only necessary if `t` is set.
        verbose : bool, optional
            Whether to print progress during normalization. If `tqdm` is
            installed, a progress bar shows. Default is True.
        """
        # convert z-stack to numpy array if dask
        z_stack = self.__check_stack(z_stack)
                   
        # set defaults if no user specified parameters
        if self.keep_original_scale:
            self.__init_scale_variables(z_stack)
        if t is not None:
            if not Intensify.check_numeric(t):
                raise ValueError(f"`t` should be numeric. Received {type(t)}.")
            if ref_idx is None:
                raise ValueError("Parameter `t` provided, but "
                                 "no reference slice provided. Set `ref_idx` "
                                 "before calling.")
            cutoff = stats.percentileofscore(z_stack[ref_idx].flatten(), t)
            thresholds = np.array([np.percentile(x.flatten(), cutoff)\
                               for x in z_stack])
        else:
            thresholds = np.array([filters.threshold_otsu(x)\
                                   for x in z_stack])

        # create empty z-stack for normalized data
        normed = np.zeros_like(z_stack, dtype='float')
        # create variables to track quantile values
        semi_quantiles = np.zeros((z_stack.shape[0], self.n_quantiles))
        p_quantiles = np.arange(0, 100, 100 / self.n_quantiles)
        # normalize along xy axis for each image.

        if self.xy_norm_:
            if verbose:
                print("Normalizing intensity values across xy-dimensions.")
            for i in Intensify.get_iterator(z_stack.shape[0], verbose):
                normed[i], thresholds[i] = self.xy_normalize(np.array(z_stack[i]),
                                                             thresholds[i])
                semi_quantiles[i, :] = np.percentile(normed[i][normed[i] < thresholds[i]],
                                                     p_quantiles)
        else:
            normed = z_stack.copy()
            for i in range(z_stack.shape[0]):
                semi_quantiles[i, :] = np.percentile(normed[i][normed[i] < thresholds[i]],
                                                     p_quantiles)
        # create variables for semi-quantile normalization
        agg_quantiles = np.percentile(semi_quantiles, 98, axis=0)
        semi_median = agg_quantiles[self.n_quantiles // 2]
        semi_max = agg_quantiles[-1]
        self.transform_ = interpolate.interp1d(np.arange(self.n_quantiles),
                                               agg_quantiles)
  
        if self.z_norm:
            if verbose:
                print("Normalizing intensity values across z-dimension.")
            for i in Intensify.get_iterator(z_stack.shape[0], verbose):
                normed[i] = self.z_normalize(normed[i], thresholds[i],
                                             semi_median, semi_max)
        return self.standardize_output(normed)

    def xy_normalize(self, img, t, smooth='savitzky-galore'):
        """
        Normalize along the XY plane.

        Parameters
        ----------
        img : np.ndarray
            Image to normalize.
        t : int, float
            Maximum background intensity in image.
        smooth : str, optional
            Method to smooth sampled background pixels. Default is
            'savitzky-galore' per original paper.

        Returns
        -------
        np.ndarray
            XY normalized image.
        """
        # select foreground pixels 
        fg = img >= t
        # replace high signal region with randomly sampled background
        # intensities
        mask = self.smooth_background(img, fg, smooth)
        mask /= mask.max()
        # divide normalize original image
        norm_xy = img / mask
        # standardize to old values
        norm_xy /= (np.median(norm_xy) / np.median(img))
        new_t = norm_xy[fg].min()
        return (norm_xy, new_t)

    def smooth_background(self, img, selected, method='savitzky-galore'):
        """
        Extract selected pixels from image. Replace with sampled and smooth
        pixels from remaining. 

        Parameters
        ----------
        img : np.ndarray
            Image to foreground extract + background smooth. 
        selected : np.ndarray
            Boolean array indicating which pixels to replace with background.
        method : str, optioanl
            Method to smooth sampled pixels with. Default is 'savitzky-galore`
            per original paper.

        Returns
        -------
        np.ndarray
            Image with extracted foreground replaced by sampled + smoothed
            background pixels 

        Raises
        ------
        ValueError
            Raises error if unknown smooth method is passed.
        """
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

    def z_normalize(self, img, t, semi_median, semi_max):
        """
        Normalize an image along the Z dimension using semi-quantile normalizaiton. 

        Parameters
        ----------
        img : np.ndarray
            Image to normalize.
        t : int, float
            Maximum background intensity.
        transform : callable
            Function to transform quantile position to intensity value.
        semi_median : int, float
            Median value of aggegrated semi-quantiles.
        semi_max : int, float
            Maximum value of aggegrated semi-quantiles.

        Returns
        -------
        np.ndarray
            Z normalized image.
        """
        (lower, upper) = np.percentile(img[img > 0], [25, 99])
        fg = img >= t
        # perform constrast stretching on foreground pixels
        img[fg] = self.constrast_stretch(img[fg], lower, upper,
                                              semi_median, semi_max)
        # quantile normalization on background pixels
        img[~fg] = self.quantile_normalization(img[~fg],
                                               self.n_quantiles)
        # re-scale foreground intensities
        img[fg] *= np.max(img[~fg]) / np.min(img[fg])
        # quantile normalization produces a decent amount of static:
        # suggested to smooth similar to xy smoothing of background
        if self.smooth_quartiles:
            mask = self.smooth_background(img, fg, 'savitzky-galore')
            img[~fg] = mask[~fg]
        return img

    def standardize_output(self, img):
        """Standardize intensity output."""
        if self.keep_original_scale:
            out = exposure.rescale_intensity(img,
                                             in_range=(img.min(), img.max()),
                                             out_range=(self.min_, self.max_))
            out = exposure.rescale_intensity(out,
                                             in_range=(0, 2 ** self.bits - 1),
                                             out_range='dtype')
        else:
            out = exposure.rescale_intensity(img)
        return out

    def constrast_stretch(self, pixels, lower, upper, semi_median, semi_max):
        """
        Constrast stretch pixels following methods outline in original paper.
        
        Parameters
        ----------
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
        ------
        np.ndarray
            Adjusted pixels
        """
        if self.stretch_method == 'intensify3d':
            out = (pixels - lower)\
                * ((semi_max - semi_median) / (upper - lower))\
                + semi_median
        else:
            out = exposure.rescale_intensity(pixels,
                                             out_range=(semi_median, semi_max))
        return out

    def quantile_normalization(self, pixels, n_quantiles):
        """
        Perform quantile normalization.
        
        Parameters
        ----------

        pixels : np.ndarray
            Values to normalize.
        n_quantiles : int
            Number of quantiles to compute.
        transform : callable
            Callable to transform quantile position to intensity value.

        Returns
        -------
        np.ndarray
            Normalized pixel intensities.
        """
        # transform pixel ranks to aggregate quantile space
        values = self.transform_(np.linspace(0, n_quantiles - 1, pixels.size))
        return values[np.argsort(pixels)]

# ---------------------------- Private Helper Functions ------------------------  
    def __check_stack(self, z_stack):
        """Check whether stack is a Dask array. Convert to numpy."""
        has_dask = False
        try:
            from dask.array import Array
            has_dask = True
        except ImportError:
            pass
        if has_dask and isinstance(z_stack, Array):
            print("Normalization with Dask arrays is not currently supported." \
                  " Converting to numpy.")
            z_stack = np.array(z_stack)
        return z_stack


    def __init_scale_variables(self, z_stack):
        """Retrieve original minimum + maximum intensity values."""
        has_dask = False
        try:
            from dask.array import Array
            has_dask = True
        except ImportError:
            pass
        if has_dask and isinstance(z_stack, Array):
            self.min_ = z_stack.min().compute()
            self.max_ = z_stack.max().compute()
        elif isinstance(z_stack, np.ndarray):
            self.min_ = z_stack.min()
            self.max_ = z_stack.max()
        else:
            raise ValueError("Expected numpy or dask array for image stack."\
                             f" Received: {type(z_stack)}")

# ----------------------------- Static Helper Functions ------------------------
    @staticmethod
    def check_numeric(value):
        """Check if value is numeric."""
        try:
            float(value)
        except:
            return False
        return True

    @staticmethod
    def get_iterator(n, verbose):
        """Get slice iterator."""
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
        """Sample pixels from a provided sample."""
        p1 = np.percentile(intensities, 10)
        p2 = np.percentile(intensities, 90)
        samples = intensities[np.logical_and(p1 < intensities,
                                             intensities < p2)]
        ecdf = ECDF(samples)
        return ecdf.x[np.searchsorted(ecdf.y, np.random.uniform(size=n))] 

    @staticmethod
    def savitzky_galoy(img, dy, dx, k):
        """Smooth an image using a Savitzky-Galoy filter."""
        by_row = signal.savgol_filter(img, dy, k, deriv=0,
                                      delta=1.0, axis=0, mode='interp',
                                      cval=0.0)
        by_col = signal.savgol_filter(by_row, dx, k, deriv=0,
                                      delta=1.0, axis=1, mode='interp',
                                      cval=0.0)
        return by_col
