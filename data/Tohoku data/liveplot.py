from time import time, sleep
import numpy as np
from queue import Empty


def live_plot(name, q, shape, xmin, xmax, xs, xlabel, ymin, ymax, ys, ylabel, prefer_2D=True, cmap='plasma'):
    import matplotlib as mpl
    #mpl.use('Qt4Agg')
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

    last_update = time()
    # Create nan-filled cache array
    last_update_index = [0] * len(shape)
    if len(shape) > 2:
        # Restrict the plot range within the last two dimensions
        shape = shape[-2:]
    data = np.full(shape, np.nan)

    fig = plt.figure()
    fig.canvas.set_window_title(name)
    ax = fig.add_subplot(111)

    make_image_plot = prefer_2D and len(shape) >= 2
    if make_image_plot:
        xmin -= xs * 0.5
        xmax += xs * 0.5
        ymin -= ys * 0.5
        ymax += ys * 0.5
        if xmin > xmax:
            xmin, xmax = xmax, xmin
            xstep = -1
        else:
            xstep = 1
        if ymin > ymax:
            ymin, ymax = ymax, ymin
            ystep = 1
        else:
            ystep = -1
        im = ax.imshow(data[::ystep, ::xstep], cmap=cmap, aspect='auto', interpolation='nearest', vmin=None, vmax=None, extent=(xmin, xmax, ymin, ymax))
        cb = plt.colorbar(im)
        cb.set_label(name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        xarr = np.linspace(xmin, xmax, shape[-1])
        if len(shape) >= 2:
            for i in np.ndindex(shape[:-1]):
                ax.plot(xarr, data[i])
        else:
            ax.plot(xarr, data)
        plt.xlabel(xlabel)
        plt.ylabel(name)

    plt.pause(0.0001)
    cfm = plt.get_current_fig_manager()
    #cfm.window.activateWindow()
    #cfm.window.raise_()
    cfm.window.attributes('-topmost', 1)
    cfm.window.attributes('-topmost', 0)
    #plt.show(False)
    #plt.draw()

    while True:
        end = False
        try:
            update = q.get(timeout=1000.0)
            if isinstance(update, tuple): # A tuple of index and new values
                index = update[0]
                if len(index) > 2:
                    if last_update_index[:-2] != index[:-2]:
                        # We have to reset the data, since higher dimension indexes changed
                        data[...] = np.nan
                last_update_index = index
                data[index[-2:] if len(index) > 2 else index] = update[1]
            else:
                end = True
            now = time()
            if last_update + 0.1 < now or end:
                last_update = now
                if make_image_plot:
                    ax.images[0].set_data(data[::ystep, ::xstep])
                    ax.images[0].set_clim(np.nanmin(data), np.nanmax(data))
                else:
                    num_lines = len(ax.lines)
                    if num_lines == 1:
                        ax.lines[0].set_ydata(data) # This is actually 1D data
                    else:
                        for j in xrange(num_lines):
                            ax.lines[j].set_ydata(data[j])
                    ax.relim()
                    ax.autoscale_view()
                ax.figure.canvas.draw()
                plt.pause(0.0001)
            if end: # Wait for a while and close the figure
                sleep(0.2)
                break
        except Empty:
            break
    plt.close(fig)
