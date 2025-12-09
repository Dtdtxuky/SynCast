import matplotlib.pyplot as plt
from client.s3_client import s3_client
import numpy as np
from visualizers.sevir_color import cmap_dict
from visualizers.shanghai_color import shanghai_cmap_dict
from visualizers.taasrad_color import taasrad_cmap_dict
import os
import argparse


dataset_list = ['sevir', 'hko7', 'TAASRAD19', 'shanghai', 'SRAD2018', 'NMIC_CAP30', 'NMIC_CR', 'MeteoNet']
model_list = ['EarthFormer', 'PredRNN',  'tau', 'incepu']

def get_parser():
    parser = argparse.ArgumentParser(description='plot_paper_figure')
    parser.add_argument('--dataset', type=str, default='sevir', choices=dataset_list)
    parser.add_argument('--model', type=str, default='EarthFormer', choices=model_list)
    args = parser.parse_args()

    assert args.dataset in dataset_list
    assert args.model in model_list

    return args


plot_dict = {
    'hko7': {
        'vmin': 0, 'vmax': 60, 'cmap': 'jet'
    },
    'sevir': cmap_dict('vil'),
    'shanghai': shanghai_cmap_dict('dbz'),
    'TAASRAD19': {
        'vmin':0, 'vmax': 55, 'cmap': 'cividis'
    },
    'SRAD2018': {
        'vmin': 0, 'vmax': 55, 'cmap': 'plasma'
    },
    'NMIC_CAP30': {
        'vmin': 0, 'vmax': 255, 'cmap': 'gist_ncar'
    },
    'NMIC_CR': {
        'vmin': 0, 'vmax': 255, 'cmap': 'gist_ncar'
    },
    'MeteoNet': {
        'vmin': 0, 'vmax': 255, 'cmap': 'gist_ncar'
    }
    # 'TAASRAD19': 
    # 'SRAD2018':
    # 'shanghai':
}

deblur_date_map = { ## Timestep12
    'sevir':{'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'hko7': {'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'shanghai': {'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'TAASRAD19': {'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'SRAD2018': {'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'NMIC_CAP30': {'EarthFormer': '0927', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'NMIC_CR': {'EarthFormer': '0927', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
    'MeteoNet': {'EarthFormer': '0925', 'PredRNN': '0925', 'tau': '0925', 'incepu': '0925'},
}

### func ###
def preprocess_data(data, dataset):
    if dataset == 'sevir':
        data = data * 255.0
        data = np.clip(data, 0, 255).squeeze()
    elif dataset == 'hko7':
        data = data * 70.0 - 10
        data = np.clip(data, 0, 70).squeeze()

        # ## pixel2mm
        # hko_zr_a = 58.53
        # hko_zr_b = 1.56
        # dbz = np.clip(data * 70.0 - 10, 0, 70).squeeze()
        # dbr = (dbz - 10.0*np.log10(hko_zr_a)) / hko_zr_b
        # rainfall_intensity = np.power(10, dbr/10.0)
        # data = rainfall_intensity
    elif dataset == 'shanghai':
        data = data.squeeze() * 70.0
    elif dataset == 'TAASRAD19':
        data = data.squeeze() * 52.5
    elif dataset == 'SRAD2018':
        data = data.squeeze() * 70.0
    # elif dataset == 'SRAD2018':
    #     data = data.squeeze() * 80.0
    elif dataset == 'NMIC_CR':
        data = data.squeeze() * 255.0
    elif dataset == 'NMIC_CAP30':
        data = data.squeeze() * 255.0
    elif dataset == 'MeteoNet':
        data = data.squeeze() * 255.0
        # import pdb; pdb.set_trace()
    # elif dataset == 'shanghai':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    # elif dataset == 'TAASRAD19':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    # elif dataset == 'SRAD2018':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    # elif dataset == 'NMIC_CR':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    # elif dataset == 'NMIC_CAP30':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    # elif dataset == 'MeteoNet':
    #     data = data * 255.0
    #     data = np.clip(data, 0, 255).squeeze()
    else:
        raise NotImplementedError
    
    return data

def get_alpha_mask(data, dataset):
    if dataset == 'sevir':
        tau = -15
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'hko7': 
        tau = -0.4
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'shanghai':
        tau = -0.4
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'TAASRAD19':
        tau = -0.1
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'SRAD2018':
        tau = -5
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'NMIC_CR':
        tau = -0.1
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'NMIC_CAP30':
        tau = -0.1
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    elif dataset == 'MeteoNet':
        tau = -0.1
        alpha = data.copy()
        alpha[alpha <= tau] = 0
        alpha[alpha > tau] = 1
    else :
        raise NotImplementedError
    return alpha

### init ###
client = s3_client(bucket_name='gongjunchao', user='jc')

### plot GDP ###
args = get_parser()
dataset = args.dataset
model = args.model
timestep = 'TimeStep12'

## read list ##
# import ast
# with open(f'data_list\\l10O12\\{dataset}\\test_12.txt', 'r') as f:
#     lines = f.readlines()
# lst_1h = ast.literal_eval(lines[0])

select_idx_map = {
    'tau': {'sevir': 6543, 'hko7': 339, 'TAASRAD19': 3118, 'shanghai': 87, 'SRAD2018': 2678,},
    'PredRNN': {'sevir': 6614, 'hko7': 794, 'TAASRAD19': 2329, 'shanghai': 1233, 'SRAD2018': 4070,},
    'incepu': {'sevir': 3661, 'hko7': 983, 'TAASRAD19': 588, 'shanghai': 829, 'SRAD2018': 2448,},
    'EarthFormer': {'sevir': 6333, 'hko7': 772, 'TAASRAD19': 2432, 'shanghai': 432, 'SRAD2018': 1155,},
}

sample_id = select_idx_map[model][dataset]

rows = 3 
assert rows == 3
lines = 1
assert lines == 1

fig, axes = plt.subplots(rows, lines, figsize=(6*lines, 6*rows))


##########################################
##########################################
# line0: plot observations #
exp_setting = 'I10O12'
ceph_path = f'radar_deblur/blur_data/{exp_setting}/{dataset}/PredRNN/{timestep}/tar_{sample_id}.npy'

plt_data = preprocess_data(client.read_npy_from_BytesIO(ceph_path), dataset)
ax = axes[0]
ax.set_yticks([])
ax.set_xticks([])

im = ax.imshow(plt_data, alpha=get_alpha_mask(plt_data, dataset), **plot_dict[dataset])


######################################
######################################
# line1: plot pred
ceph_path = f'radar_deblur/blur_data/{exp_setting}/{dataset}/{model}/{timestep}/pred_{sample_id}.npy'
plt_data = preprocess_data(client.read_npy_from_BytesIO(ceph_path), dataset)

### plot ###
ax = axes[1]
ax.set_yticks([])
ax.set_xticks([])

im = ax.imshow(plt_data, alpha=get_alpha_mask(plt_data, dataset), **plot_dict[dataset])

######################################
######################################
# line2: plot deblur
deblur_date = deblur_date_map[dataset][model]
ceph_path = f'radar_deblur/deblur_data/new_GDP/{dataset}/{model}/{timestep}/{deblur_date}/{sample_id}.png'
plt_data = preprocess_data(client.read_png_from_BytesIO(ceph_path), dataset)

### plot ###
ax = axes[2]
ax.set_yticks([])
ax.set_xticks([])

im = ax.imshow(plt_data, alpha=get_alpha_mask(plt_data, dataset), **plot_dict[dataset])


dataset2title_map = {'sevir': 'SEVIR', 'hko7': 'HKO7', 'shanghai': 'Shanghai', 'TAASRAD19': 'TAASRAD19', 'SRAD2018': 'SRAD2018',}
dataset_title = dataset2title_map[dataset]
fig.suptitle(f'{dataset_title}', fontsize=20)
cb_title_map = {'sevir': 'VIL(pixel)', 'hko7': 'DBZ(dBZ)', 'shanghai': 'DBZ(dBZ)', 'TAASRAD19': 'DBZ(dBZ)', 'SRAD2018': 'DBZ(dBZ)',
                'NMIC_CAP30': 'DBZ(dBZ)', 'NMIC_CR': 'DBZ(dBZ)', 'MeteoNet': 'DBZ(dBZ)'}

font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }

cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.08, pad=0.04, shrink=1, extend='max',
                )
cb_title = cb_title_map[dataset]
cb.ax.set_title(f'{cb_title}', fontdict=font)
cb.ax.tick_params(labelsize=16)

plt.subplots_adjust(left=0., right=1.0, bottom=0.2, top=0.965, wspace=0.05, hspace=0.05)

# plt.tight_layout()
# plt.savefig(f'{dataset}\\{model}_{dataset}_{sample_id}.png', bbox_inches='tight', pad_inches=0)
save_dir = os.path.join('paper_plt', model)
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f'{dataset}_{sample_id}.png'), bbox_inches='tight', pad_inches=0.01)