from copy import deepcopy
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib import colors


MeteoNet_CMAP =  colors.ListedColormap(['lavender','indigo','mediumblue','dodgerblue', 'skyblue','cyan',
                          'olivedrab','lime','greenyellow','orange','red','magenta','pink'])

def get_norm():
    borne_max = 56 + 10
    bounds = [0,4,8,12,16,20,24,32,40,48,56,borne_max]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=MeteoNet_CMAP.N)
    return norm 

def cmap_dict():
    return {
        'cmap': MeteoNet_CMAP,
        'norm': get_norm(),
        # 'vmin': 0,
        # 'vmax': 60
    }
