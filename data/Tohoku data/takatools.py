# -*- coding: utf-8 -*-
"""
Collection of convenient functions for analysis and plotting

Created on Sat Aug 24 22:41:55 2013
Last update: May 22, 2016 by Taka

@author: Takashi Nakajima
"""
from __future__ import division

import sys
from math import *
from collections import OrderedDict
import numpy as np
import xarray as xr
from scipy import ndimage

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import Button, MultiCursor, Slider
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.tight_bbox import adjust_bbox

try:
    import weakref
    import ipywidgets as ip
    from IPython.display import display, clear_output
    from traitlets import *

    import lmfit
    from lmfit import Parameter

    import io
    #from cStringIO import StringIO
    from PIL import Image
    import win32clipboard
except:
    pass


def new_array(orig_xarr, replaced_dim=None, new_dim=None, new_coords=None, name=None, dtype=None, fill_zeros=False):
    """Create a new empty xarray.DataArray with the same data structure as orig_xarr.
    If replaced_dim is given, the specified dimension is replaced with the new
    dimension name new_dim and the corresponding coordinates array new_coords."""
    shape = list(orig_xarr.shape)
    if replaced_dim is not None:
        if not isinstance(replaced_dim, int):
            # Extract dimension index to be replaced
            for i, dim_name in enumerate(orig_xarr.dims):
                if dim_name == replaced_dim:
                    replaced_dim = i
                    break
        if new_dim is None:
            del shape[replaced_dim]  # Just remove this dimension
        else:
            shape[replaced_dim] = len(new_coords)
    if dtype is None:
        dtype = orig_xarr.dtype
    num_array = np.zeros(shape, dtype=dtype) if fill_zeros else np.empty(shape, dtype=dtype)

    if isinstance(orig_xarr, xr.DataArray):
        # Copy coordinate information
        coords = []
        for i, dim_name in enumerate(orig_xarr.dims):
            if replaced_dim is not None and dim_name == orig_xarr.dims[replaced_dim]:
                if new_dim is not None:
                    coords.append((new_dim, new_coords))
            else:
                coords.append((dim_name, orig_xarr.coords[dim_name]))
        return xr.DataArray(num_array, coords, name=name, attrs=orig_xarr.attrs)
    else:
        return num_array


def axis(dataset, dim):
    """Obtain an dimension scale associated to an xarray DataArray or an HDF5 dataset if exists"""
    try:
        if dim < 0:
            ndim = len(dataset.shape)
            dim += ndim
        if isinstance(dataset, xr.DataArray):
            name = dataset.dims[dim]
            arr = dataset.coords[name]
            units = dataset.attrs.get('units', '')
            if units:
                name += ' (' + units + ')'
        else: # HDF5 dataset
            name = list(dataset.dims[dim].keys())[0]
            arr = dataset.dims[dim][0]
    except:
        name = ""
        arr = None
    return name, arr


def slice_for_range(data, fromVal, toVal):
    """
    Returns a slice indicating the region [fromVal, toVal] in a 1D ndarray data

    Example:
    >>> data = linspace(1, 50, 50)
    >>> s = slice_for_range(data, 4, 10)
    >>> data[s]
    array([  4.,   5.,   6.,   7.,   8.,   9.,  10.])
    """
    if (data[-1] - data[0]) / (toVal - fromVal) < 0:
        # Order of fromVal and toVal is reversed
        temp = toVal
        toVal = fromVal
        fromVal = temp

    foundFromVal = False
    start = None
    stop = None
    if toVal > fromVal:
        # ascending order
        for idx, val in enumerate(data):
            if foundFromVal:
                if val > toVal:
                    stop = idx
                    break
            else:
                if val >= fromVal:
                    start = idx
                    foundFromVal = True
    else:
        # descending order
        for idx, val in enumerate(data):
            if foundFromVal:
                if val < toVal:
                    stop = idx
                    break
            else:
                if val <= fromVal:
                    start = idx
                    foundFromVal = True

    return slice(start, stop)


def mesh(arr):
    """
    Generate mesh array appropriate for use with pyplot.pcolor and pyplot.pcolormesh
    """
    delta = arr[1] - arr[0]
    return np.append(arr, arr[-1] + delta) - delta/2.0


def differentiate(xarr, dim=1, order=1, sigma=1, mode='reflect'):
    """
    Compute finite difference by Gaussian convolution.
    Note that the differentiation of multiple dimensions (dim > 1) here means the multiple of
    differential coefficients for each dimension.

    dim is the dimension of the differentiation counted from the last index,
    that is equal to the dimension of arr by default.
    order is the order of the differentiation.
    sigma is the standard deviation of the Gaussian kernel: larger sigma gives more smoothing.

    Theoretical background:
        Since (f*g)' = f'*g = f*g' where * denotes convolution, the convolution of
        the differentiation f' and Gaussian g equals to the convolution of f and the
        differentiation of Gaussian g.
        Thus, differentiation + Gaussian smoothing can be replaced with the convolution with
        the derivative of Gaussian.

        ref. http://stackoverflow.com/questions/18991408/python-finite-difference-functions
    """
    result = xarr.copy()

    for idx in np.ndindex(xarr.shape[:-dim]):
        result[idx] = ndimage.gaussian_filter(xarr[idx], sigma=sigma, order=order, mode=mode)
    return result


def cummean(xarr, axis=-1, dtype=None):
    """
    Deprecated. Use boxcar() instead.
    Calculate normalized version of the cumulative sum of xarr
    """
    return boxcar(xarr, axis, dtype)

def boxcar(xarr, axis=-1, dtype=None):
    """
    Calculate normalized version of the boxcar integration of xarr
    """
    norm = np.linspace(1, xarr.shape[axis], xarr.shape[axis])
    return np.cumsum(xarr, axis=axis, dtype=dtype) / norm


def dual_gaussian_model(norm=False):
    def dual_gaussian(x, x1, x2, a1, a2, sigma):
        gauss = lambda m, s, x: np.exp(-(x - m)**2 / (2 * s**2))
        return a1 * gauss(x1, sigma, x) + a2 * gauss(x2, sigma, x)

    def norm_dual_gaussian(x, x1, x2, r, sigma):
        gauss = lambda m, s, x: 1.0 / (s * sqrt(2 * pi)) * np.exp(-(x - m)**2 / (2 * s**2))
        return r * gauss(x1, sigma, x) + (1.0 - r) * gauss(x2, sigma, x)

    model = lmfit.Model(norm_dual_gaussian if norm else dual_gaussian, independent_vars=['x'])
    return model


def damped_oscillation_model():
    def damped_oscillation(x, f, phi, t2, t_drift, a_osc, a_drift, a_offset):
        return a_osc / 2.0 * (1.0 + np.cos(2 * pi * f * x + phi) * np.exp(-(x / t2)**2)) + a_drift * np.exp(-x / t_drift) + a_offset

    model = lmfit.Model(damped_oscillation, independent_vars=['x'])
    return model


def gaussians_to_population(y_array, x_array, xl, xr):
    """
    Fit y_array with two Gaussian distributions with the initial guess of their center
    coordinates xl and xr.
    Returns a Parameter object representing the ratio of the left population.

    Example:
        left_populations = data.new_data('Left_populations', -1)
        for idx in np.ndindex(left_populations.shape):
            left_populations[idx] = gaussian_to_population(data['Digitizer_count'][idx], data['Volt'], 0.12, 0.16).value
    """
    def dual_gaussian(x, ratio, sigma, center_left, sep):
        gauss = lambda m, s, x: 1.0/(s*sqrt(2*pi)) * np.exp(-(x-m)**2/(2*s**2))
        return ratio * gauss(center_left, sigma, x) + (1.0-ratio) * gauss(center_left + sep, sigma, x)

    # Normalize distribution so that the integration by x gives one
    norm = y_array.sum() * (x_array[1] - x_array[0])

    model = lmfit.Model(dual_gaussian, independent_vars=['x'])
    result = model.fit(y_array / norm, x=x_array,
                    ratio=Parameter(value=0.5, min=0.0, max=1.0),
                    sigma=Parameter(value=(xr-xl)/2, max=(xr-xl)),
                    center_left=xl,
                    sep=Parameter(value=xr-xl, min=(xr-xl)/2))
    return result.params['ratio']


def singlet_probabilities(dist_arrays, x_array, sep, x_pos=None):
    """
    Fit n-dimensional array of the double Gaussian distribution dist_arrays
    and returns (n-1)-dimensional singlet-return probabilities by thresholding.
    The probability of the peak positioned at x_pos is returned.
    If x_pos is None (default), the initial guess is given by np.argmax of the first histogram.

    Example:
        ps = data.new_data('Singlet_probabilities', -1)
        ps[:] = singlet_probabilities(data['Digitizer_count'], data['Volt'], 0.010)
    """
    integral_dist = np.sum(dist_arrays, axis=tuple(range(len(dist_arrays.shape)-1)), dtype=np.float64)
    out_shape = dist_arrays.shape[:-1]

    def dual_gaussian(x, ratio, sigma, x_pos, sep):
        gauss = lambda m, s, x: 1.0/(s*sqrt(2*pi)) * np.exp(-(x-m)**2/(2*s**2))
        return ratio * gauss(x_pos, sigma, x) + (1.0-ratio) * gauss(x_pos + sep, sigma, x)

    # Normalize distribution so that the integration by x gives one
    full_count = integral_dist.sum()
    integral_dist = integral_dist / (full_count * (x_array[1] - x_array[0]))

    model = lmfit.Model(dual_gaussian, independent_vars=['x'])
    result = model.fit(integral_dist, x=x_array,
                    ratio=Parameter(value=0.5, min=0.0, max=1.0),
                    sigma=Parameter(value=abs(sep)/2, max=abs(sep)),
                    x_pos=x_pos,
                    sep=Parameter(value=sep, min=sep*(2**copysign(1, -sep)), max=sep*(2**copysign(1, sep))))

    threshold = result.params['x_pos'].value + result.params['sep'].value / 2
    singlet_slice = slice_for_range(x_array, x_array[0], threshold)

    count = full_count / np.product(out_shape)
    ps = np.empty(out_shape, dtype=np.float64)
    for idx in np.ndindex(out_shape):
        ps[idx] = np.sum(dist_arrays[idx][singlet_slice], dtype=np.float64) / count
    return ps

def find_thresholds(xarr, bins=None, xdrift=None, ydrift=None):
    """
    Find threshold values from binary scattered data or histrogram in xarr.
    Systematic drift of the threshold value is guessed automatically.

    :param array-like xarr: scattered value data or histogram data in xr.DataArray or other array-like types
    :param int,string bins: (required for histogram data) axis index or dimension name for bin values in histogram data
    :param int,string xdrift: (optional) axis index or dimension name along which the threshold drift should be guessed
    :param int,string ydrift: (optional) 2nd axis index or dimension name along which the threshold drift should be guessed
    :rtype: xr.DataArray
    :return: array of threshold values with the dimensions of ydrift and xdrift axes
    """
    def residual(pars, x=None, y=None, bins=None, data=None):
        # unpack parameters:
        #  extract .value attribute for each parameter
        parvals = pars.valuesdict()
        
        model = parvals['a0']
        if x is not None:
            model += parvals['a1'] * x + parvals['a2'] * x * x
        if y is not None:
            model += parvals['b1'] * y + parvals['b2'] * y * y
        if data is None:
            # Evaluate model values with the given parameters
            try:
                # Successful only when model is an instance of xarray
                model.name = 'thresholds'
            except:
                pass
            return model
        
        if bins is None:
            distances = np.abs(data - model)
        else:  # histogram data
            distances = np.abs(bins - model) * data
        m = np.mean(distances)
        return np.ravel(distances - m)

    xarr = xr.DataArray(xarr)
    kws = {'data':xarr}
    params = lmfit.Parameters()
    if bins is None:
        params.add('a0', np.median(xarr))
    else:
        bins = bins if bins in xarr.coords else xarr.dims[bins]
        kws['bins'] = xarr.coords[bins]
        params.add('a0', np.mean(xarr.coords[bins] * xarr))
    if xdrift is not None:
        params.add('a1', 0.)
        params.add('a2', 0.)
        xdrift = xdrift if xdrift in xarr.coords else xarr.dims[xdrift]
        kws['x'] = xarr.coords[xdrift]
    if ydrift is not None:
        params.add('b1', 0.)
        params.add('b2', 0.)
        ydrift = ydrift if ydrift in xarr.coords else xarr.dims[ydrift]
        kws['y'] = xarr.coords[ydrift]

    result = lmfit.minimize(residual, params, kws=kws)
    kws.pop('data')
    return residual(result.params, **kws)

def probability_lower(xarr, thresholds, axis=None, bins=None):
    """
    Calculates the probability of values lower than thresholds.
    xarr is assumed to be histogram data if bins is given, otherwise it is assumed to be scattered value data.

    :param array-like xarr: scattered value data or histogram data in xr.DataArray or other array-like types
    :param array-like thresholds: threshold values
    :param int,string axis: (optional) axis index or dimension name from which ensemble of data is taken to calculate the probability
    :param int,string bins: (optional) axis index or dimension name for histogram bins
    """
    xarr = xr.DataArray(xarr)
    if bins is None:
        axis = axis if axis in xarr.coords else xarr.dims[axis]
        count = len(xarr.coords[axis])
        count_array = (xarr < thresholds).sum(dim=axis)
    else:
        bins = bins if bins in xarr.coords else xarr.dims[bins]
        count = xarr.sum(dim=bins)
        count_array = ((xarr.coords[bins] < thresholds) * xarr).sum(dim=bins)

    prob = new_array(count_array, dtype=np.float32)
    prob[...] = count_array
    prob /= count
    return prob

"""
def find_threshold(xarr, sep, left_index=None, drift=True):
    """"""
    Fit n-dimensional array of the double Gaussian distribution xarr
    and returns (n-1)-dimensional array of threshold values.
    If drift is True, a quadratic drift of the center of mass is assumed along
    the second last axis and it is removed by fitting.
    If left_index is None (default), the initial guess is given by np.argmax
    and left_index is assumed to be (the guessed value) - sep if it is larger
    than the center of mass.
    """"""
    # Setup fitting
    def dual_gaussian(x, ratio, sigma, x_pos, sep):
        gauss = lambda m, s, x: 1.0/(s*sqrt(2*pi)) * np.exp(-(x-m)**2/(2*s**2))
        return ratio * gauss(x_pos, sigma, x) + (1.0-ratio) * gauss(x_pos + sep, sigma, x)
    model = lmfit.Model(dual_gaussian, independent_vars=['x'])
    separation = float(abs(sep))

    drift_model = lmfit.models.QuadraticModel()

    # Setup buffers
    out_shape = xarr.shape[:-1]
    thresholds = new_array(xarr, -1, dtype=np.float64)
    dist_buff = np.empty(xarr.shape[-1], dtype=np.float64)
    t_buff = np.empty(xarr.shape[-2], dtype=np.int32)
    x_array = np.arange(xarr.shape[-1])
    t_array = np.arange(xarr.shape[-2])

    try:
        # Get the backend numpy array if xarr is an xarray.DataArray instance
        dist_arrays = xarr.values
    except:
        dist_arrays = xarr

    guessed_separations = []

    for idx in np.ndindex(out_shape[:-1]):
        for j in xrange(out_shape[-1]):
            np.cumsum(dist_arrays[idx][j], out=dist_buff)
            for center_of_mass in xrange(len(dist_buff)):
                if dist_buff[center_of_mass] >= dist_buff[-1]/2:
                    t_buff[j] = center_of_mass
                    break
        # Guess the slow drift of x_pos
        drift_fit = drift_model.fit(t_buff, x=t_array, a=0.0, b=0.0, c=t_buff[0]).params
        if drift:
            dist_buff[:] = 0
            shift = drift_fit['a'].value * t_array**2 + drift_fit['b'].value * t_array
            for j in xrange(out_shape[-1]):
                dist_buff += np.roll(dist_arrays[idx][j], -int(round(shift[j])))
        else:
            dist_buff = np.sum(dist_arrays[idx], axis=0)
            shift = 0
        # Make the initial guess for double Gaussian fitting
        init_index = np.argmax(dist_buff) if left_index is None else left_index
        if init_index > drift_fit['c'].value + separation/2:
            init_index -= separation
        # Perform fitting
        fit = model.fit(dist_buff/dist_buff.sum(), x=x_array,
                        ratio=Parameter(value=0.5, min=0.0, max=1.0),
                        sigma=Parameter(value=separation/2, max=separation*2),
                        x_pos=float(init_index),
                        sep=Parameter(value=separation, min=separation/2, max=separation*2)).params
        thresholds[idx] = fit['x_pos'].value + shift
        if 0.1 < fit['ratio'].value < 0.9:
            # Assume that the value of the separation was estimated correctly only if
            # the populations of the two Gaussian distributions is not too much different
            guessed_separations.append(fit['sep'].value)
    if len(guessed_separations) > 0:
        thresholds += np.average(guessed_separations) / 2
    return thresholds


def threshold_left(xarr, thresholds, index_thresholds=False, normalize=False):
    """"""
    Thresholding the histogram data in xarr by thresholds
    and returns the population (probability) on the left side
    (population of the values lower than thresholds).
    """"""
    thresholds = np.asarray(thresholds)
    out_shape = xarr.shape[:-1]
    if out_shape != thresholds.shape:
        if thresholds.shape:
            raise Exception("Shape of thresholds is inconsistent with the given histogram.")
        else:
            # Assume thresholds is a scalar value
            thresholds = np.full(out_shape, thresholds)

    probabilities_xarr = new_array(xarr, -1, dtype=np.float32)
    if hasattr(xarr, 'coords') and not index_thresholds:
        bin_edges = xarr.coords[xarr.dims[-1]].values # This is the array of the bins' center positions
        bin_step = bin_edges[1] - bin_edges[0]
    else:
        bin_edges = np.arange(xarr.shape[-1], dtype=np.float32)
        bin_step = 1.0
    bin_edges -= bin_step / 2 # Assume homogeneous bin widths

    # Cast xarr to np.ndarray to speed up data access
    xarr = np.asarray(xarr)
    probabilities = np.asarray(probabilities_xarr)
    thresholds[...] = (thresholds - bin_edges[0]) / bin_step
    for idx in np.ndindex(out_shape):
        th = thresholds[idx]
        probabilities[idx] = np.sum(xarr[idx][:int(th)]) + xarr[idx][int(th)] * (th - int(th))
    if normalize:
        # Distribution has to be normalized to represent probabilities
        sum = np.sum(xarr[(0,) * len(out_shape)])
        probabilities /= sum
    return probabilities_xarr
"""

def relaxation_time(y_array, x_array):
    """
    Fit y_array with exponential decay function.
    Returns a Parameter object representing the relaxation time.
    """
    def exponential_decay(x, y0, amplitude, time):
        return y0 + amplitude * np.exp(-x/time)

    model = lmfit.Model(exponential_decay, independent_vars=['x'])
    result = model.fit(y_array, x=x_array, y0=y_array[-1], amplitude=(y_array[0]-y_array[-1]), time=(x_array[-1] - x_array[0])/2)
    return result.params['time']


def fit_decaying_oscillation(y_array, x_array, d=2, freq_init=None, amp_init=None, phase_init=0, y0_init=None):
    """
    Fit y_array with Gaussian decay oscillations.
    """
    def oscillation(t, y0, amplitude, freq, t2, phase):
        return y0 + amplitude * np.exp(-(t/t2)**d) * np.cos(2*pi*freq*t + phase)

    time_scale = x_array[-1] - x_array[0]
    if freq_init is None:
        freq_init = 10.0/time_scale
    if amp_init is None:
        amp_init = y_array[0] - y_array[-1]
    if y0_init is None:
        y0_init = y_array[-1]
    model = lmfit.Model(oscillation, independent_vars=['t'])
    result = model.fit(y_array, t=x_array, y0=y0_init, amplitude=amp_init, freq=freq_init, t2=Parameter(value=time_scale/2, min=0), phase=phase_init)
    return result


def remove_plane(xarr, x_range=None, y_range=None):
    """
    Remove plane background from 2D data.
    xarr is a multidimensional array of ndim >= 2.
    x_range and y_range are slices for x-axis and y-axis where the slope of the plane is calculated.

    Example:
        flattened = remove_plane(xarr, (xarr.P3 < -0.3) & (xarr.P3 > -0.5), (xarr.P1 > -0.5) & (xarr.P1 < -0.2))
    """
    if x_range is None:
        x_range = slice(None)
    if y_range is None:
        y_range = slice(None)

    grad_y, grad_x = np.gradient(xarr[..., y_range, x_range])
    result = xarr.copy()

    x = np.arange(xarr.shape[-1])
    y = np.arange(xarr.shape[-2])
    xx, yy = np.meshgrid(x, y)

    for idx in np.ndindex(xarr.shape[:-2]):
        result[idx] -= xarr[idx][0][0].values + np.median(grad_x[idx]) * xx + np.median(grad_y[idx]) * yy

    return result


def remove_vertical_noise(xarr):
    """
    Remove vertical oscillations in 2D data.
    Useful to remove the fluctuation of the background signal appearing sweep by sweep.
    """
    result = xarr.copy()

    fft2D = np.fft.fft2(xarr)
    fft2D[..., 0] = 0
    result[...] = np.real(np.fft.ifft2(fft2D))
    return result


def boxcar_hist(arr, bins=256, range=None):
    """
    Deprecated. Use boxcar() and hist() instead.
    Performs normalized boxcar integration along the last axis of n-dim arr and makes histograms with bins and range
    """
    if range is None:
        range = (np.min(arr), np.max(arr))

    int_len = arr.shape[-1]
    int_buffer = np.zeros(np.prod(arr.shape[:-1]), dtype=np.float64)

    output = np.empty((int_len, bins), dtype=np.float32)

    for int_idx in xrange(int_len):
        int_buffer += arr[..., int_idx]
        output[int_idx], bin_edges = np.histogram(int_buffer / (int_idx + 1), bins=bins, range=range, density=True)

    return output


def hist(xarr, axis=0, bins=100, range=None):
    """
    Calculate histogram of the elements in the specified axis
    """
    if range is None:
        m = float(np.mean(xarr))
        std = float(np.std(xarr, ddof=1))
        range = (m-std*3, m+std*3)
    step = (range[-1] - range[0]) / (bins-1)
    bins_array = np.linspace(range[0], range[-1], bins)
    result = new_array(xarr, replaced_dim=axis, new_dim="bins",
                        new_coords=bins_array, name="histogram", dtype=np.uint32)

    shape = list(xarr.shape)
    shape[axis] = 1
    for idx in np.ndindex(tuple(shape)):
        i = list(idx)
        i[axis] = slice(None)
        i = tuple(i)
        result[i], _ = np.histogram(xarr[i], bins=np.linspace(range[0] - step/2, range[-1] + step/2, bins+1), density=False)

    return result


def bayesian(data_k, p_jk, prior_distribution=None):
    # Update probability distribution using Bayes' rule, i.e., multiply p_jk if data_k == 1 else (1-p_jk)
    # We assume that data_k is binary data or probability between 0 and 1
    prior_distribution = np.ones(p_jk.shape[0]) if prior_distribution is None else prior_distribution
    updated = prior_distribution * np.prod(((1. - p_jk) + data_k * (2. * p_jk - 1)), axis=-1)
    return updated / np.sum(updated)


"""
Plotting functions
"""


def implot(data, *args, **kwargs):
    """
    Convenient function to make a 2D image plot.
    args can be either x and y arrays or a set of xmin, xmax, ymin and ymax.

    Example:
        tk.implot(data, x_array, y_array)
        tk.implot(data, x_array.min(), x_array.max(), y_array.min(), y_array.max(), axes=ax)
    """
    axes = kwargs.pop('axes', plt.gca())
    im = axes.imshow(data, **kwargs)

    if args is not None:
        if len(args) == 2:
            im.set_extent((args[0][0], args[0][-1], args[1][0], args[1][-1]))
        elif len(args) == 4:
            im.set_extent(args)
    return im

def make_segments(array_list):
    """
    Create list of line segments from n-dim coordinates [x,y,z,...], in the correct
    format for LineCollection: an array of the form numlines * (points per line) * n
    """

    points = np.asarray(array_list).T.reshape(-1, 1, len(array_list))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_colorline(x, y, z=None, cmap='jet', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, **kwargs):
    """
    http://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x)-1)

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = [z]
    z = np.asarray(z)

    segments = make_segments([x, y])
    ax = plt.gca()
    cm = plt.get_cmap(cmap)

    if len(kwargs) > 0:
        # Use Line2Ds for more flexible control of appearance
        lines = []
        for i in xrange(len(segments)):
            aline = Line2D(segments[i,:,0], segments[i,:,1], alpha=alpha, linewidth=linewidth,
                           color=cm(z[min(i,len(z)-1)]), **kwargs)
            ax.add_line(aline)
            lines.append(aline)
        return lines
    else:
        # Use LineCollection for best performance
        lc = LineCollection(segments, array=z, cmap=cm, norm=norm, linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc


def plot(dset, **kwargs):
    defaults = {'name': '', 'show_comment': False, 'show_title': True, 'auto_copy': True,
                'cmap': 'plasma', 'reverse': False, 'auto_scale': True,
                'offset': 0., 'xlog': False, 'ylog': False}
    defaults.update(kwargs)

    if isinstance(dset, xr.DataArray):
        dset = xr.Dataset({dset.name:dset})

    # We must use a weak reference to hdf so that it can be released properly
    # when not used, since otherwise ipywidgets.interact captures all the arguments.
    dset_ref = weakref.ref(dset)

    data_keys = list(dset.data_vars.keys())
    name = defaults['name']
    if name not in data_keys:
        name = data_keys[-1]
    name_widget = ip.Dropdown(options=data_keys, description='name', value=name)
    show_comment = ip.ToggleButton(description='show note', value=defaults['show_comment'])
    show_title = ip.Checkbox(description='show title', value=defaults['show_title'])
    auto_copy = ip.Checkbox(description='auto copy', value=defaults['auto_copy'])
    comment_widget = ip.HTML(value='<pre><code>{0}</code></pre>'.format(dset.attrs.get("comment", '')))
    comment_widget.layout.display = 'none'

    def __show_comment_changed(change):
        comment_widget.layout.display = '' if show_comment.value else 'none'
    __show_comment_changed(None)
    show_comment.observe(__show_comment_changed, 'value')

    copy_position_button = ip.Button(description='Copy [x, y]') #Tomo
    copy_distance_button = ip.Button(description='Copy [dx, dy]') #Tomo
    copy_position_button.layout.width = copy_distance_button.layout.width = '300px'
    def __copy_val(btn):
        try:
            text = str(btn.value).decode('utf8')
            text = text.replace('(','[').replace(')',']') #Tomo
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            win32clipboard.CloseClipboard()
        except:  # Ignore errors
            pass
    copy_distance_button.on_click(__copy_val)
    copy_position_button.on_click(__copy_val)

    sub_widgets = []
    plot_widgets = []
    plot_fig = []

    def _plot_2D(data, facet_dict, fname, data_label):
        maps = [cm for cm in plt.colormaps() if not cm.endswith("_r")]
        for k in ('magma', 'inferno', 'plasma', 'viridis'):
            maps.remove(k)
            maps.insert(0, k)
        #maps.sort()
        cmap_widget = ip.Dropdown(description='cmap', options=maps, value=defaults['cmap'])
        reverse_widget = ip.Checkbox(description='reverse', value=defaults['reverse'])

        robust_widget = ip.Checkbox(description='auto scale', value=defaults['auto_scale'])
        a = np.ravel(data)
        min_v, max_v = np.nanmin(a[a != -np.inf]), np.nanmax(a[a != np.inf])
        scale_widget = ip.FloatRangeSlider(min=min_v, max=max_v, step=(max_v-min_v)/100., value=(min_v, max_v), continuous_update=False)
        scale_widget.layout.visibility = 'hidden'

        def __robust_changed(change):
            scale_widget.layout.visibility = 'hidden' if robust_widget.value else 'visible' #Tomo
        robust_widget.observe(__robust_changed, 'value')

        def __replot(change):
            cmap = cmap_widget.value
            if reverse_widget.value:
                cmap += '_r'
            if not facet_dict:
                plt.figure(figsize=(6,4.5))
            if robust_widget.value:
                data.plot(cmap=cmap, robust=True, **facet_dict)
            else:
                min_v, max_v = scale_widget.value
                print (min_v, max_v)
                data.plot(cmap=cmap, vmin=min_v, vmax=max_v, **facet_dict)
            plt.gca().collections[-1].colorbar.set_label(data_label)

        ws = [cmap_widget, reverse_widget, robust_widget, scale_widget]
        return ws, __replot

    def _plot_1D(data_list, fname, data_label):
        offset_widget = ip.FloatText(description='offset', value=defaults['offset'])

        def __replot(change):
            offset = offset_widget.value
            for i in range(len(data_list)):
                (data_list[i] + offset * i).plot()
            plt.ylabel(data_label)

        ws = [offset_widget]
        return ws, __replot

    def _update_plot(change):
        if plot_widgets:
            plot_widgets.pop().close()
        dset = dset_ref()
        facets = []
        idx = []
        for c in sub_widgets[0].children:
            if c.value == 'facet':
                facets.append(len(idx))
                idx.append(slice(None))
            else:
                idx.append(c.value)
        data = dset[name_widget.value][tuple(idx)]
        faceted_dims = len(data.shape) - len(facets)
        if faceted_dims > 2 and len(facets) < 2:
            for i in range(len(data.shape)):
                if i not in facets:
                    facets.append(i)
                    faceted_dims -= 1
                    if faceted_dims == 2 or len(facets) == 2:
                        break

        try:
            units = dset[name_widget.value].units
            data_label = '{0} ({1})'.format(name_widget.value, units)
        except:
            data_label = name_widget.value
        fname = dset.attrs.get('filename', '')
        if faceted_dims == 2:  # 2D plot
            facet_dict = {}
            if facets:
                facet_dict['col'] = data.dims[facets[-1]]
                if len(facets) == 2:
                    facet_dict['row'] = data.dims[facets[0]]
            widgets, plot_func = _plot_2D(data, facet_dict, fname, data_label)
        else:  # Make stacked line plots
            s = list(data.shape)
            i = len(s) - 1
            while i > 0:
                if i not in facets:
                    break
                i -= 1
            s[i] = 1
            dl = []
            for idx in np.ndindex(tuple(s)):
                j = list(idx)
                j[i] = slice(None)
                dl.append(data[tuple(j)])
            widgets, plot_func = _plot_1D(dl, fname, data_label)

        xscale_widget = ip.Checkbox(description='xlog', value=defaults['xlog'])
        yscale_widget = ip.Checkbox(description='ylog', value=defaults['ylog'])
        padding_widget = ip.FloatSlider(description='pad', min=0., max=1.0, step=0.1, value=0.2, continuous_update=False)
        widgets.extend([xscale_widget, yscale_widget, padding_widget])

        def __plot_wrapper(change):
            if plot_fig:
                plt.close(plot_fig.pop())
            clear_output(wait=True)
            display(comment_widget)
            display(ip.HBox([name_widget, show_comment, show_title, auto_copy]))
            display(sub_widgets[0])
            display(plot_widgets[0])

            plot_func(change)

            fig = plt.gcf()
            plot_fig.append(fig)
            if show_title.value:
                fig.suptitle(fname, fontsize=16, y=1.02)
            # Workaround for tha lack of png transparency support in Windows
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)

            if xscale_widget.value:
                plt.gca().set_xscale('log')
            if yscale_widget.value:
                plt.gca().set_yscale('log')

            # Adjust axis formatters
            for ax in fig.axes:
                for xyax in [ax.xaxis, ax.yaxis]:
                    if type(xyax.get_major_formatter()) == ScalarFormatter:
                        fmt = ScalarFormatter(useMathText=True, useOffset=False)
                        fmt.set_powerlimits((-3, 3))
                        xyax.set_major_formatter(fmt)

            cid = []
            prev_clicked = [0, 0]
            display(ip.HBox([copy_position_button, copy_distance_button]))

            def __adjust(event):
                # Adjust figure padding to accomodate the suptitle and axis labels
                # Call this only once after the figure is displayed
                fig.canvas.mpl_disconnect(cid[0])
                bb = fig.bbox_inches.padded(padding_widget.value)
                adjust_bbox(fig, bb)
                fig.set_size_inches(bb.bounds[2], bb.bounds[3], forward=True)
                if auto_copy.value:
                    copy2clipboard(fig, pad_inches=padding_widget.value)

            def __clicked(event):
                copy_position_button.value = (event.xdata, event.ydata)
                copy_position_button.description = 'Copy [x,y]=[%f,%f]' % copy_position_button.value #Tomo
                copy_distance_button.value = (event.xdata - prev_clicked[0], event.ydata - prev_clicked[1])
                copy_distance_button.description = 'Copy [dx,dy]=[%f,%f]' % copy_distance_button.value #Tomo
                prev_clicked[0] = event.xdata
                prev_clicked[1] = event.ydata
                
            cid.append(fig.canvas.mpl_connect('draw_event', __adjust))
            fig.canvas.mpl_connect('button_press_event', __clicked)

        for w in widgets:
            w.observe(__plot_wrapper, 'value')
        plot_widgets.append(ip.HBox(widgets))
        __plot_wrapper(None)

    def _update(change):
        if sub_widgets:
            sub_widgets.pop().close()

        dset = dset_ref()
        index_widgets = []
        for dim in dset[name_widget.value].dims:
            coord = dset[name_widget.value].coords[dim]
            indexes = OrderedDict({':': slice(None)})
            indexes['facet'] = 'facet'
            for i in range(len(coord)):
                indexes[float(coord[i])] = i
            drop = ip.Dropdown(description=dim, options=indexes)
            drop.observe(_update_plot, 'value')
            index_widgets.append(drop)

        sub_widgets.append(ip.HBox(index_widgets))
        _update_plot(None)

    for w in [name_widget, show_title]:
        w.observe(_update, 'value')
    _update(None)


def copy2clipboard(fig=None, **kwargs):
    '''
    copy a matplotlib figure to clipboard as BMP on windows
    http://stackoverflow.com/questions/7050448/write-image-to-windows-clipboard-in-python-with-pil-and-win32clipboard
    '''
    if not fig:
        fig = plt.gcf()
    defaults = {'format': 'png', 'facecolor': 'w', 'transparent': False, 'dpi': 'figure'}
    defaults.update(kwargs)

    try:
        buf = io.BytesIO()
        #fig.savefig(buf, format="png", bbox_inches='tight')
        fig.savefig(buf, **defaults)
        image = Image.open(buf)

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()

        output = StringIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()

        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)  # DIB = device independent bitmap
        win32clipboard.CloseClipboard()

        buf.close()
    except:  # Ignore errors
        pass


def inspect(data, X=None, Y=None, cmap='gray', hcolor='cyan'):
    viewers = []
    def __do_plot(cmap, hcolor):
        while len(viewers) > 0:
            d = viewers.pop()
            d.close()
        d = DataViewer(data, X, Y)
        d.plot(cmap=cmap, hcolor=hcolor)
        viewers.append(d)
    cmaps = sorted(plt.cm.datad.keys())
    ipywidgets.interact(__do_plot, cmap=ipywidgets.Dropdown(options=cmaps, value=cmap),
                    hcolor=ipywidgets.Dropdown(options=['black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white'], value=hcolor))


class DataViewer:
    """Displays 2D data for simple inspection.
    A new instance has to be recreated if figure window is closed.
    """
    def __init__(self, data, X=None, Y=None):
        self.data = data
        self.X = np.arange(self.data.shape[1]) if X is None else X
        self.Y = np.arange(self.data.shape[0]) if Y is None else Y

        vmin = np.min(self.data)
        vmax = np.max(self.data)

        # Create and displays the figure object.
        self.fig = plt.figure(figsize=(8,8), frameon=True, tight_layout=True)

        # Create grid for layout
        grid = GridSpec(4,4)

        self.ax_main = self.fig.add_subplot(grid[0:3,0:3])
        #self.ax_main.autoscale(enable=True, tight=True)
        self.ax_main.autoscale(enable=False)
        self.ax_main.set_xlim(np.min(self.X), np.max(self.X))
        self.ax_main.set_ylim(np.min(self.Y), np.max(self.Y))
        # Use 'auto' to adjust the aspect ratio to fill the figure window, 'equal' to fix it.
        self.ax_main.set_aspect('auto', adjustable='box-forced')

        self.ax_h = self.fig.add_subplot(grid[3,0:3], sharex=self.ax_main)
        self.ax_h.set_axis_bgcolor('0.8')
        self.ax_h.autoscale(False)
        self.ax_h.set_ylim(vmin, vmax)

        self.ax_v = self.fig.add_subplot(grid[0:3,3], sharey=self.ax_main)
        self.ax_v.set_axis_bgcolor('0.8')
        self.ax_v.autoscale(False)
        self.ax_v.set_xlim(vmax, vmin)

        self.prev_pt = None
        self.ax_cb = None

        self.cursor = MultiCursor(self.fig.canvas, (self.ax_main, self.ax_h, self.ax_v),
                                  horizOn=True, vertOn=True, color='white', ls='--', lw=1)
        self.fig.canvas.mpl_connect('button_press_event', self._plot_clicked)

        # Setup control buttons
        btn_grid = GridSpecFromSubplotSpec(4, 1, subplot_spec=grid[3,3])
        self.btn_colorbar = Button(self.fig.add_subplot(btn_grid[2,0]), 'Colorbar')
        self.btn_colorbar.on_clicked(self._plot_colorbar)
        self.btn_reset = Button(self.fig.add_subplot(btn_grid[3,0]), 'Reset')
        self.btn_reset.on_clicked(self._plot_reset)

        # Setup color range sliders
        self.slider_vmin = Slider(self.fig.add_subplot(btn_grid[0,0]), "vmin", vmin, vmax, valinit=vmin)
        self.slider_vmin.on_changed(self._plot_rangechanged)
        self.slider_vmax = Slider(self.fig.add_subplot(btn_grid[1,0]), "vmax", vmin, vmax, valinit=vmax, slidermin=self.slider_vmin)
        self.slider_vmax.on_changed(self._plot_rangechanged)
        self.slider_vmin.slidermax = self.slider_vmax

        self.fig.canvas.draw()

    def close(self):
        plt.close(self.fig)

    def plot(self, title=None, xlabel=None, ylabel=None, cmap='gray', hcolor='cyan'):
        self.cmap = cmap
        self.hcolor = hcolor
        if title is not None:
            self.ax_main.set_title(title)
        if xlabel is not None:
            self.ax_main.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax_main.set_ylabel(ylabel)

        vmin = self.slider_vmin.val
        vmax = self.slider_vmax.val
        self.ax_main.collections = []
        self.colormesh = self.ax_main.pcolormesh(mesh(self.X), mesh(self.Y), self.data, cmap=self.cmap, vmin=vmin, vmax=vmax)
        self.ax_h.set_ylim(vmin, vmax)
        self.ax_v.set_xlim(vmax, vmin)
        if self.ax_cb is None:
            self.fig.canvas.draw()
        else:
            # Update colorbar
            self.fig.delaxes(self.ax_cb)
            self.ax_cb = None
            self._plot_colorbar(None)

    def _plot_rangechanged(self, event):
        self.plot(cmap=self.cmap, hcolor=self.hcolor)

    def _plot_colorbar(self, event):
        if self.ax_cb is None:
            self.ax_cb = self.fig.add_axes([0.6,0.7,0.02,0.2])
            self.colorbar = self.fig.colorbar(self.colormesh, cax=self.ax_cb, orientation='vertical')
            self.colorbar.outline.set_edgecolor(self.hcolor)
            self.ax_cb.yaxis.set_tick_params(color=self.hcolor)
            for label in self.ax_cb.get_yticklabels():
                label.set_color(self.hcolor)
        else:
            self.fig.delaxes(self.ax_cb)
            self.ax_cb = None
        self.fig.canvas.draw()

    def _plot_reset(self, event):
        self.ax_main.lines = []
        self.ax_main.texts = []
        self.ax_h.lines = []
        self.ax_v.lines = []
        self.prev_pt = None
        print("Reset")
        sys.stdout.flush()
        self.fig.canvas.draw()

    def _plot_clicked(self, event):
        if event.inaxes == self.ax_main:
            # Get nearest data
            xpos = np.argmin(np.abs(event.xdata - self.X))
            ypos = np.argmin(np.abs(event.ydata - self.Y))
            click_pt = (self.X[xpos], self.Y[ypos])
            if self.prev_pt is not None:
                # Draw an arrow
                self.ax_main.annotate('', xy=click_pt,  xycoords='data', xytext=self.prev_pt, textcoords='data',
                                      arrowprops=dict(arrowstyle="->", connectionstyle="arc3", ec=self.hcolor, shrinkA=4, shrinkB=4))
                delta = (click_pt[0]-self.prev_pt[0], click_pt[1]-self.prev_pt[1])
                print("Delta({:g},{:g})".format(*delta))
                sys.stdout.flush()
            self.prev_pt = click_pt
            # Check which mouse button:
            if event.button == 3: # right click
                # Plot cross sections
                c, = self.ax_h.plot(self.X, self.data[ypos, :], label="{:.3g}".format(click_pt[1]))
                self.ax_main.axhline(click_pt[1], color=c.get_color(), ls='-', lw=1)
                c, = self.ax_v.plot(self.data[:, xpos], self.Y, label="{:.3g}".format(click_pt[0]))
                self.ax_main.axvline(click_pt[0], color=c.get_color(), ls='-', lw=1)
            elif event.button == 1: # left click
                self.ax_main.plot([click_pt[0]], [click_pt[1]], marker='o', markerfacecolor=self.hcolor)
            self.ax_main.annotate("{:.3g},{:.3g}".format(*click_pt), xy=click_pt,
                                  xytext=(4.0,4.0), textcoords='offset points', color=self.hcolor, size='small', stretch='condensed')
            print("Pt({:g},{:g})".format(*click_pt))
            sys.stdout.flush()
            self.fig.canvas.draw()
