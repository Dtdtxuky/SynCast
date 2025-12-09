"""Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License."""

from copy import deepcopy
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt


VIL_COLORS = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]

VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]

def get_cmap(type, encoded=True):
    if type.lower() == 'vis':
        cmap, norm = vis_cmap(encoded)
        vmin, vmax = (0, 10000) if encoded else (0, 1)
    elif type.lower() == 'vil':
        cmap, norm = vil_cmap(encoded)
        vmin, vmax = None, None
    elif type.lower() == 'ir069':
        cmap, norm = c09_cmap(encoded)
        vmin, vmax = (-8000, -1000) if encoded else (-80, -10)
    elif type.lower() == 'lght':
        cmap, norm = 'hot', None
        vmin, vmax = 0, 5
    else:
        cmap, norm = 'jet', None
        vmin, vmax = (-7000, 2000) if encoded else (-70, 20)
    return cmap, norm, vmin, vmax


def vil_cmap(encoded=True):
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    # Exactly the same error occurs in the original implementation (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/src/display/display.py).
    # ValueError: There are 10 color bins including extensions, but ncolors = 9; ncolors must equal or exceed the number of bins
    # We can not replicate the visualization in notebook (https://github.com/MIT-AI-Accelerator/neurips-2020-sevir/blob/master/notebooks/AnalyzeNowcast.ipynb) without error.
    nil = cols.pop(0)
    under = cols[0]
    # over = cols.pop()
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    return cmap, norm

def cmap_dict(s):
    return {'cmap': get_cmap(s, encoded=True)[0],
            'norm': get_cmap(s, encoded=True)[1],
            'vmin': get_cmap(s, encoded=True)[2],
            'vmax': get_cmap(s, encoded=True)[3]}


def plot_single_image(data):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(data, **cmap_dict('vil'))

    plt.show()
    return None