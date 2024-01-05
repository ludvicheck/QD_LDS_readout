# -*- coding: utf-8 -*-
"""
Created on Jan 10 2016
Last Updated on Aug 30, 2016

!! For live plotting with Python 2.7.10 on Windows, you have to fix the following bug in multiprocessing module. !!
Open C:¥Python27¥Lib¥multiprocessing¥forking.py and replace the line #478:
===
if main_name != 'ipython':
===
with the following code:
===
if main_name == '__main__':
    main_module = sys.modules['__main__']
    main_module.__file__ = main_path
elif main_name != 'ipython':
===
.

@author: Takashi Nakajima
"""

import numpy as np
from netCDF4 import Dataset
import xarray
import re, os, sys
from datetime import datetime
from IPython import get_ipython
#import weakref
import warnings
from multiprocessing import Process, Queue
import liveplot

try:
    # For Python 2.7, accepts bytes as text
    from StringIO import StringIO
except:
    # For Python 3.x, accepts only unicode string
    from io import StringIO

# def plot(dset, plot_2D=True, cmap='plasma', robust=True):
# Plot functions have been removed from this file.
# Please use takatools.plot() instead.


def load_data(filename="", load_now=False):
    """Load xarray.Dataset from an existing netCDF4 file (.h5) or a legacy .dat data file.
    Data are loaded lazily from a disk if load_now is False, but loaded entirely into memory if load_now is True.
    Note that when you modify values of a Dataset, only the in-memory copy you are manipulating in xarray is modified:
    the original file on disk is never touched.

    A loaded Dataset has to be closed by invoking close() method, or use a context manager:
        with load_data() as d:
            print(d['data'])
            ...
    """
    try:
        fnum = int(filename)
    except:
        pass
    else:
        # If only an integer number is given, assume it is an HDF5 file
        filename = str(filename) + '.h5'

    if not filename:
        filelist = os.listdir('.')
        filelist.sort(reverse=True)
        for f in filelist:
            if re.match(r'.*?\.(dat|h5)$', f):
                filename = f
                break

    if re.match(r'.*?\.h5$', filename):
        # This is a netCDF4 file
        data = xarray.open_dataset(filename)
        if load_now:
            data.load()
        # Make aliases of coordinates to take into account their units
        swapped_keys = {}
        for coord_key in data.coords:
            coord = data.coords[coord_key]
            try:
                units = coord.units
                if units is not None:
                    new_key = '{key} ({units})'.format(key=coord_key, units=units)
                    swapped_keys[coord_key] = new_key
                    data.coords[new_key] = (coord_key, coord)
            except:
                pass
        data = data.swap_dims(swapped_keys)
    elif re.match(r'(.*?\d+)(_\d+)*\.dat$', filename):
        # This is a legacy data file from FDAQ
        # Create an in-memory Dataset representation
        def __loadDimensionFromLegacyFile(storage, basename, subshape, meas_names, first_param_ascending, indices):
            if len(subshape) == 1:
                # Last dimension. Load 1D data from file
                for i in indices:
                    basename = basename + '_' + str(i)
                try:
                    # 'U' means the universal newline mode, where any kind of newline
                    # symbols are replaced with '\n'.
                    # This is the default behavior in Python 3
                    with open(basename + '.dat', 'rU') as f:
                        head, data = f.read().split('\n\n')
                    # Check sweep direction
                    xVals = np.loadtxt(StringIO(data), dtype=np.float64, usecols=(0,))
                    needs_reverse = (first_param_ascending != (xVals[-1] - xVals[0] > 0))
                    for i in range(len(meas_names)):
                        # Load i-th measured variable
                        data = np.loadtxt(StringIO(data), dtype=np.float64, usecols=(i+1,))
                        if needs_reverse:
                            data[:] = data[::-1]
                        # Indices must be tuple! (list indices are for "fancy indexing")
                        storage[meas_names[i]][tuple(indices)] = data
                except Exception as e:
                    # Ignore any error while reading data from files
                    print(e)
            else:
                # Loop dimension index
                for i in range(subshape[0]):
                    new_indices = indices[:]
                    new_indices.append(i)
                    __loadDimensionFromLegacyFile(storage, basename, subshape[1:], meas_names, first_param_ascending, new_indices)

        shape = []
        param_names = []
        meas_names = []
        first_param_ascending = True
        data = xarray.Dataset()

        with open(filename, 'rU') as f:
            # Keep header information
            header = f.read().split('\n\n')[0]
        # Save the header information as an ASCII string attribute of the file
        data.attrs["comment"] = unicode(header)

        stepPat = re.compile(r'Step\d+\s+(\S+):\s+([0-9eE\.+-]+)\s+-\s+([0-9eE\.+-]+)\s+.*\((\d+)\s*points\)')
        measPat = re.compile(r'Meas:(\S+\s+)+')

        header_lines = header.split('\n')
        j = 0
        # Read moving parameters
        for j in range(len(header_lines)):
            m = stepPat.match(header_lines[j])
            if m:
                try:
                    pname = m.group(1)
                    i = 0
                    while pname in param_names:
                        pname = m.group(1) + str(i)
                        i += 1
                    param_names.append(pname)
                    # Create a new dimension scale
                    data.coords[pname] = ((pname,), np.linspace(float(m.group(2)), float(m.group(3)),float(m.group(4))))
                    shape.append(int(m.group(4)))
                    if j == 0 and float(m.group(2)) > float(m.group(3)):
                        first_param_ascending = False
                except ValueError:
                    print("Error in interpreting header")
            else:
                break

        shape.reverse()
        param_names.reverse()

        # Read measured variables
        m = measPat.match(header_lines[j])
        if m:
            try:
                for name in m.groups():
                    meas_names.append(name.rstrip())
                    # Prepare storage array
                    data[name.rstrip()] = (tuple(param_names), np.empty(shape, np.float32))
            except ValueError:
                print("Error in reading data")

        # Let's auto-load all of the relevant data files
        m = re.match(r'(.*?\d+)(_\d+)*\.dat$', filename)
        __loadDimensionFromLegacyFile(data, m.group(1), shape, meas_names, first_param_ascending, [])
    else:
        data = None

    return data


class data_store(object):
    """
    @params
    filename (string): Filename without a file extention
    comment (string): Arbitrary comment text
    diskless (bool): If True (default), only an in-memory representation is initially created.
                    If False, any change is immediately written out to disk.
    persist (bool): If True (default), the in-memory data is written out to disk when _file is closed.
                    Effective only when diskless=True.
    weakref (bool): From netCDF4 description:
                    If True, child Dimension and Variable instances will keep
                    weak references to the parent Dataset or Group object.
                    Default is False, which means strong references will be kept.
                    Having Dimension and Variable instances keep a strong reference to
                    the parent Dataset instance, which in turn keeps a reference to
                    child Dimension and Variable instances, creates circular references.
                    Circular references complicate garbage collection, which may mean
                    increased memory usage for programs that create many Dataset instances
                    with lots of Variables. Setting keepweakref=True allows Dataset
                    instances to be garbage collected as soon as they go out of scope,
                    potential reducing memory usage. However, in most cases this is
                    not desirable, since the associated Variable instances may still
                    be needed, but are rendered unusable when the parent Dataset instance
                    is garbage collected.
    format (string): Data format of the file, which is "NETCDF4" by default.
    """
    def __init__(self, filename="", comment="", diskless=True, persist=True, weakref=False, format="NETCDF4"):
        if not filename:
            filename = datetime.now().strftime('%y%m%d%H%M%S')
        #Create file, fail if exists
        self._file = Dataset(filename + '.h5', mode="w", clobber=False, diskless=diskless, persist=persist, weakref=weakref, format=format)
        self._file.filename = filename
        self.setup_plot(live_plot=diskless)
        if not diskless:
            print("In-memory caching is disabled and the data will be written out directly to disk. Live plotting is not available in this mode.")

        if not comment:
            try:
                from instr import world
                comment = world.status
            except:
                pass
        try:
            # Save user execution command if used inside ipython
            ip = get_ipython()
            if comment:
                comment += "\n\n"
            comment += ip.history_manager.input_hist_raw[ip.execution_count]
        except:
            pass

        if comment:
            self._file.comment = unicode(comment)

        self._reverse_loops = set()
        self._idx = [] # Current index of running loops
        self._plot_processes = {} # For live plotting
        self._live_queues = {} # Update queues of the acquired data for plotting
        self.axis_names = []
        self.meas_names = []

    def __del__(self):
        if self._file is not None:
            print("Closed {0}".format(self._file.filename))
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._file is not None:
            for q in self._live_queues.values():
                q.put('end')
            self._live_queues = {}
            self._plot_processes = {}
            filename = self._file.filename + '.h5'
            self._file.close()
            self._file = None

            if value is None:  # No error occurred
                # Reopen the file as an xarray dataset
                print("Reopened {0} in read-only mode".format(filename))
                self._file = load_data(filename)

    def __getitem__(self, key):
        return self._file[key]

    def __setitem__(self, key, value):
        self._file[key] = value

    def __getattr__(self, attr):
        if hasattr(self, '_file'):
            return getattr(self._file, attr)
        else:
            raise AttributeError()

    @property
    def idx(self):
        return tuple(self._idx)

    def axis(self, pts_or_arr, first=0, last=None, var=None, units=None, dim=None, compression=True):
        """Add a new axis for data and variable loops"""
        if not hasattr(pts_or_arr, '__iter__'):
            if not pts_or_arr > 0:
                raise Exception("Number of points must be a positive integer")
            if last is None:
                last = first + pts_or_arr - 1
            pts_or_arr = np.linspace(first, last, int(round(pts_or_arr)))

        if var is None:
            label = "axis {0}".format(len(self.axes))
        else:
            try:
                # var might be an instr.variable instance
                label = var.name
                # limit check
                allowed_min, allowed_max = var.min, var.max
                if allowed_min > min(pts_or_arr) or allowed_max < max(pts_or_arr):
                    raise Exception("Destination value of {0} is out of range ({1}, {2})".format(label, allowed_min, allowed_max))
                if units is None:
                    try:
                        units = var.units
                    except:
                        pass
            except AttributeError:
                label = var

        if dim is None:
            dim = len(self.axis_names)
        elif not 0 <= dim <= len(self.axis_names):
            raise Exception("New axis dimension must be {0} or smaller".format(len(self.axis_names)))
        # Register the created dataset as an axis dataset
        if dim == len(self.axis_names):
            self._file.createDimension(label, len(pts_or_arr))
            self.axis_names.append([label])
            dim_name = label
        else:
            self.axis_names[dim].append(label)
            dim_name = self.axis_names[dim][0]
        # Reset idx with correct dimensions
        self._idx = [slice(None)] * len(self.axis_names)
        ax = self._file.createVariable(label, pts_or_arr.dtype, (dim_name,), zlib=compression)
        ax[:] = pts_or_arr

        if units is not None:
            ax.units = units

        return ax

    def loop(self, dim=None, reverse=False):
        """Generator to loop over axis values at a given dimension"""
        default_slice = slice(None)
        if dim is None:
            # If dim is not given, defaults to the outer-most dimension not looped over yet
            for i in xrange(len(self.axis_names)):
                if self._idx[i] == default_slice:
                    dim = i
                    break
            else:
                raise Exception("Loop dimension is too deep. data_store.axis() must be called as many times as the number of loop dimensions.")
        else:
            dim %= len(self.axis_names)
        axes = [self[name] for name in self.axis_names[dim]]
        should_reverse = False
        if reverse:
            if dim in self._reverse_loops:
                # Forward loop was running previously. Reverse the loop direction.
                self._reverse_loops.remove(dim)
                should_reverse = True
            else:
                # Backward loop was running previously.
                self._reverse_loops.add(dim)
        xr = xrange(len(axes[0]) - 1, -1, -1) if should_reverse else xrange(len(axes[0]))
        def __loop_generator():
            for i in xr:
                self._idx[dim] = i
                yield tuple(ax[i] for ax in axes) if len(axes) > 1 else axes[0][i]
            # Reset to the default slice when exiting from the current loop
            self._idx[dim] = default_slice
        return __loop_generator()

    def put(self, value, name=None, index=None, print_update=True):
        """Set value to an existing dataset, self[name][index].
        When name is None, value is assigned to the most recently created dataset.
        When index is None, value is assigned at the index of currently running loops.
        """
        name = self.meas_names[-1] if name is None else name
        index = self.idx if index is None else index
        self._file[name][index] = value
        self._file.sync()  # Write out data
        if print_update:
            sys.stdout.write("\r{0} done.".format(index))
            sys.stdout.flush()
        if self._live_plot and name in self._live_queues:
            if name not in self._plot_processes:
                self._setup_live_plot(name)
            self._live_queues[name].put((index, value))

    def storage(self, meas=None, units=None, dim=0, dtype=np.float32, compression=True, chunksizes=None, comment="", noplot=False):
        """Create a new data storage array with specified name and dimension dim.
        If dim is a tuple of axis names, a new array with those axis dimensions is created.
        If dim == 0, the dimensions will be the same as the pre-defined axes.
        If dim < 0, the number of dimensions will be smaller than the pre-defined axes by dim.
        """
        if meas is None:
            name = 'data'
        else:
            try:
                # meas might be an instr.measurable instance
                name = meas.name
                if units is None:
                    try:
                        units = meas.units
                    except:
                        pass
            except AttributeError:
                name = meas

        if isinstance(dim, tuple):
            dims = dim
        else:
            if not self.axis_names:
                raise Exception("At least one axis must be defined before creating a new dataset")
            dims = tuple(axes[0] for axes in self.axis_names)
            if dim != 0:
                dims = dims[:dim]

        # Adjust chunk and cache sizes automatically
        # This has critical influence on the performance
        max_cache_size = 2**28  # 256MB
        shape = tuple([len(self._file.dimensions[dim_name]) for dim_name in dims])
        data_size = np.prod(shape) * np.dtype(dtype).itemsize
        # Create a cache large enough to accomodate all the data if possible
        # Default cache size is 2**22 = 4MB
        cache_size = min(max(data_size, 2**22), max_cache_size)
        if chunksizes is None:
            chunksizes = list(shape)
            i = 0
            while i < len(shape) - 1 and np.prod(chunksizes) * np.dtype(dtype).itemsize > cache_size / 2:
                # Reduce the chunk size from the outer-most dimension
                if chunksizes[i] >= 2:
                    chunksizes[i] = int(chunksizes[i]/2)
                else:
                    i += 1
        storage = self._file.createVariable(name, dtype, dims, zlib=compression, chunksizes=chunksizes, fill_value=np.nan)
        storage.set_var_chunk_cache(size=cache_size)

        if self._live_plot:
            if np.prod(shape) > 20000000:
                warnings.warn("Live plotting is disabled because a dataset of size > 20Mpts is created. It is recommended to call data_store() with diskless=False to prevent on-memory caching.")
                # Disable data caching for live plotting because it is too large.
                self._live_plot = False
            elif not noplot:
                self._live_queues[name] = Queue()
        # Register the created dataset as a measurement dataset
        self.meas_names.append(name)
        if units is not None:
            storage.units = units
        if comment:
            storage.comment = unicode(comment)

        return storage

    def _setup_live_plot(self, name):
        shape = self._file[name].shape
        xarr = self._file[self._file[name].dimensions[-1]]
        dx = xarr[1]-xarr[0] if len(xarr) > 1 else 0
        if self._plot_2D and len(shape) >= 2:
            yarr = self._file[self._file[name].dimensions[-2]]
            dy = yarr[1]-yarr[0] if len(yarr) > 1 else 0
            p = Process(target=liveplot.live_plot, args=(name, self._live_queues[name], shape,
                    xarr[0], xarr[-1], dx, xarr.name,
                    yarr[0], yarr[-1], dy, yarr.name,
                    self._plot_2D, self._default_cmap))
        else:
            p = Process(target=liveplot.live_plot, args=(name, self._live_queues[name], shape,
                    xarr[0], xarr[-1], dx, xarr.name,
                    0, 1, 1, '',
                    self._plot_2D, self._default_cmap))

        self._plot_processes[name] = p
        p.start()

    def setup_plot(self, live_plot=True, plot_2D=True, cmap='plasma'):
        self._live_plot = live_plot
        self._plot_2D = plot_2D
        self._default_cmap = cmap