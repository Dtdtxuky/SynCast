if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')

from utils.misc import get_rank, get_world_size, is_dist_avail_and_initialized
import numpy as np
from megatron_utils import mpu
import matplotlib.pyplot as plt
import os
import io

from typing import Optional, Sequence, Union, Dict
import math
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib import colors
from copy import deepcopy


try:
    from petrel_client.client import Client
except:
    pass



from copy import deepcopy
from matplotlib.colors import ListedColormap, BoundaryNorm

# 配色与阈值
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

def vil_cmap(encoded=True):
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    nil = cols.pop(0)
    under = cols[0]
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    return cmap, norm


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
    
class non_visualizer(object):
    pass



class meteonet_visualizer(object):
    def __init__(self, exp_dir, sub_dir='meteonet_train_vis'):
        self.exp_dir = exp_dir
        # self.hko_zr_a = 58.53
        # self.hko_zr_b = 1.56
        self.sub_dir = sub_dir
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'{self.exp_dir}/{sub_dir}_{timestamp}'
        os.makedirs(self.save_dir, exist_ok=True)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 70 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 70

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 70
                val_min = 0
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                print(f'{self.save_dir}/pixel_step{step}_time{t}.png')
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()

    def save_pixel_image_4(self, pred_image, target_img, ref, step, batchidx, predcsi, predfar, refcsi, reffar):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy()  # b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 70 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :, 0] * 70 # to pixel

            ref_imgs = ref.detach().cpu().numpy()
            pxl_ref_imgs = ref_imgs[0, :, 0] * 70 # to pixel

            fig, axes = plt.subplots(3, pxl_pred_imgs.shape[0], figsize=(36, 4))  # 3 rows (GT, Pred, Ref), T columns
            val_max = 70
            val_min = 0
            for t in range(pxl_pred_imgs.shape[0]):
                # GT
                ax_gt = axes[0, t]
                ax_gt.imshow(pxl_target_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_gt.set_title(f"GT-{t}", fontsize=6)
                ax_gt.axis('off')

                # Prediction
                ax_pred = axes[1, t]
                ax_pred.imshow(pxl_pred_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_pred.set_title(f"Pred-{t}", fontsize=6)
                ax_pred.axis('off')
                ax_pred.text(2, 10, f"CSI: {predcsi:.2f}\nFAR: {predfar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

                # Reference
                ax_ref = axes[2, t]
                ax_ref.imshow(pxl_ref_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_ref.set_title(f"Ref-{t}", fontsize=6)
                ax_ref.axis('off')
                ax_ref.text(2, 10, f"CSI: {refcsi:.2f}\nFAR: {reffar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f'saved in {self.save_dir}/epoch_{step}_batchidx_{batchidx}.png')
            
    def save_meteo_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 70 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 70

        ### define color map ###
        cmap = colors.ListedColormap(['lavender','indigo','mediumblue','dodgerblue', 'skyblue','cyan',
                          'olivedrab','lime','greenyellow','orange','red','magenta','pink'])
        # Reflectivity : colorbar definition
        if (np.max(pxl_target_imgs) > 56):
            borne_max = np.max(pxl_target_imgs)
        else:
            borne_max = 56 + 10
        bounds = [0,4,8,12,16,20,24,32,40,48,56,borne_max]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, cmap=cmap, norm=norm)
        ax1.set_title(f'pred_meteo_step{step}_60min')
        im2 = ax2.imshow(last_target_img, cmap=cmap, norm=norm)
        ax2.set_title(f'target_meteo_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/meteo_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)

    def save_pixel_image_6methods(self, vis_args, step, batchidx, file_name="vis"):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            methods = ["tar", "pred", "ref", "far", "win", "lose"]
            csi_keys = ["", "csi_pred", "csi_ref", "csi_far", "csi_win", "csi_lose"]
            far_keys = ["", "far_pred", "far_ref", "far_far", "far_win", "far_lose"]

            # 转 numpy
            np_imgs = {}
            for m in methods:
                arr = vis_args[m].detach().cpu().numpy()  # b,t,c,h,w
                np_imgs[m] = arr[0, :, 0] * 70

            # 取等间隔6帧
            total_frames = np_imgs["tar"].shape[0]
            frame_ids = [0, 4, 8, 12, 16, 17]

            # fig, axes = plt.subplots(len(methods), len(frame_ids), figsize=(18, 12))  # 6x6
            # cmap, norm = vil_cmap()

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 12),
                constrained_layout=False  # 我们手动调节间距
            )
            cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            # 减小行列间距
            plt.subplots_adjust(
                wspace=0.05,  # 列间距（越小越紧凑）
                hspace=0.05   # 行间距
            )


            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    im = ax.imshow(np_imgs[m][t], cmap=cmap, norm=norm)  # ✅ 保存 im
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")

                    if col == len(frame_ids) - 1 and csi_keys[row] != "":
                        csi_val = vis_args[csi_keys[row]]
                        far_val = vis_args[far_keys[row]]
                        ax.text(
                            1.05, 0.5, f"CSI: {csi_val:.2f}\nFAR: {far_val:.2f}",
                            transform=ax.transAxes,
                            fontsize=14, color="black", va="center", ha="left",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3")
                        )

                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

                # 给每行添加 colorbar，使用最后一帧的 im
                # fig.colorbar(im, ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.04)


            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")

    def save_pred_gt_images(self, pred, gt, step=0, batchidx=0, file_name="vis"):
        """
        可视化预测(pred)与真实(gt)的多帧对比。
        pred, gt: Tensor [b, t, 1, h, w]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            methods = ["gt", "pred"]
            # 转 numpy 并归一化到 0-70
            np_imgs = {}
            np_imgs["gt"] = gt.detach().cpu().numpy()[0, :, 0] * 70  # [t,h,w]
            np_imgs["pred"] = pred.detach().cpu().numpy()[0, :, 0] * 70

            total_frames = np_imgs["gt"].shape[0]
            frame_ids = [0, 4, 8, 12, 16, min(17, total_frames-1)]  # 等间隔6帧，避免越界

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 6),  # 两行6列
                constrained_layout=False
            )

            cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            # 调整行列间距
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    im = ax.imshow(np_imgs[m][t], cmap=cmap, norm=norm)  # 保存 im 对象
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")

                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

            # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)  
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")


    def save_input_images(self, input_image, step=0, batchidx=0, file_name="vis_input"):
        """
        绘制 input 的所有 6 帧图像并保存，带 colorbar
        input_image: Tensor [b, t, c, h, w]，t=6
        """
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            arr = input_image.detach().cpu().numpy()  # b,t,c,h,w
            np_imgs = arr[0, :, 0] * 70  # 取第一个 batch，第一个通道
            total_frames = np_imgs.shape[0]  # 应该是 6

            fig, axes = plt.subplots(1, total_frames, figsize=(12, 3))
            cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            ims = []
            for col in range(total_frames):
                ax = axes[col]
                im = ax.imshow(np_imgs[col], cmap=cmap, norm=norm)
                ims.append(im)
                ax.set_title(f"t={col+1}", fontsize=10)  # 用 1-based 编号
                ax.axis("off")

            # 单独加一个 colorbar axes，避免重叠
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(ims[0], cax=cbar_ax)
            cbar.set_label("Pixel Value", fontsize=10)

            plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9)  # 留出右边位置
            out_file = os.path.join(
                self.save_dir, f"{file_name[0]}_input_epoch_{step}_batchidx_{batchidx}.png"
            )
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"saved in {out_file}")
                      
class sevir_visualizer(object):
    def __init__(self, exp_dir, sub_dir='sevir_train_vis'):
        self.exp_dir = exp_dir
        # self.hko_zr_a = 58.53
        # self.hko_zr_b = 1.56
        self.cmap_color = 'gist_ncar'
        self.sub_dir = sub_dir
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'{self.exp_dir}/{sub_dir}_{timestamp}'
        os.makedirs(self.save_dir, exist_ok=True)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ## b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 255
                val_min = 0
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                print(f'{self.save_dir}/pixel_step{step}_time{t}.png')
                
                save_path = f'{self.save_dir}/win_lose_pixel_step{step}_time{t}.png'
                
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
    
                plt.savefig(f'{self.save_dir}/win_lose_pixel_step{step}_time{t}.png', dpi=100, bbox_inches='tight', pad_inches=0)
                # print(f'saved in {{self.save_dir}/win_lose_pixel_step{step}_time{t}.png}')
                plt.clf()

    def save_pixel_image_2(self, pred_image, target_img, step, batchidx):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ## b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            fig, axes = plt.subplots(2, pred_imgs.shape[0], figsize=(36, 4))  # 2行18列

            cmap, norm = vil_cmap()
    
            for t in range(pred_imgs.shape[0]):
                ax_gt = axes[0, t]
                ax_gt.imshow(pxl_target_imgs[t], cmap=cmap, norm=norm)
                ax_gt.set_title(f"GT-{t}", fontsize=6)
                ax_gt.axis('off')

                # Prediction
                ax_pred = axes[1, t]
                ax_pred.imshow(pxl_pred_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"Pred-{t}", fontsize=6)
                ax_pred.axis('off')
                
            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f'saved in {self.save_dir}/epoch_{step}_batchidx_{batchidx}.png')


    def save_pixel_image_3(self, pred_image, target_img, ref, win, lose, step, batchidx):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ## b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            ref_imgs = ref.detach().cpu().numpy()
            pxl_ref_imgs = ref_imgs[0, :,  0] * 255

            win_imgs = win.detach().cpu().numpy()
            pxl_win_imgs = win_imgs[0, :,  0] * 255
            
            lose_imgs = lose.detach().cpu().numpy()
            pxl_lose_imgs = lose_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs
            ref_imgs = pxl_ref_imgs 
            win_imgs = pxl_win_imgs
            lose_imgs = pxl_lose_imgs
            
            fig, axes = plt.subplots(5, pred_imgs.shape[0], figsize=(36, 4))  # 2行18列

            cmap, norm = vil_cmap()
    
            for t in range(pred_imgs.shape[0]):
                ax_gt = axes[0, t]
                ax_gt.imshow(pxl_target_imgs[t], cmap=cmap, norm=norm)
                ax_gt.set_title(f"GT-{t}", fontsize=6)
                ax_gt.axis('off')

                # Prediction
                ax_pred = axes[1, t]
                ax_pred.imshow(pxl_pred_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"Pred-{t}", fontsize=6)
                ax_pred.axis('off')

                ax_pred = axes[2, t]
                ax_pred.imshow(pxl_ref_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"ref-{t}", fontsize=6)
                ax_pred.axis('off')

                ax_pred = axes[3, t]
                ax_pred.imshow(pxl_win_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"win-{t}", fontsize=6)
                ax_pred.axis('off')

                ax_pred = axes[4, t]
                ax_pred.imshow(pxl_lose_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"lose-{t}", fontsize=6)
                ax_pred.axis('off')
                
            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f'saved in {self.save_dir}/epoch_{step}_batchidx_{batchidx}.png')
            

    def save_pixel_image_4(self, pred_image, target_img, ref, step, batchidx, predcsi, predfar, refcsi, reffar):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy()  # b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255  # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :, 0] * 255

            ref_imgs = ref.detach().cpu().numpy()
            pxl_ref_imgs = ref_imgs[0, :, 0] * 255

            fig, axes = plt.subplots(3, pxl_pred_imgs.shape[0], figsize=(36, 4))  # 3 rows (GT, Pred, Ref), T columns
            cmap, norm = vil_cmap()

            for t in range(pxl_pred_imgs.shape[0]):
                # GT
                ax_gt = axes[0, t]
                ax_gt.imshow(pxl_target_imgs[t], cmap=cmap, norm=norm)
                ax_gt.set_title(f"GT-{t}", fontsize=6)
                ax_gt.axis('off')

                # Prediction
                ax_pred = axes[1, t]
                ax_pred.imshow(pxl_pred_imgs[t], cmap=cmap, norm=norm)
                ax_pred.set_title(f"Pred-{t}", fontsize=6)
                ax_pred.axis('off')
                ax_pred.text(2, 10, f"CSI: {predcsi:.2f}\nFAR: {predfar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

                # Reference
                ax_ref = axes[2, t]
                ax_ref.imshow(pxl_ref_imgs[t], cmap=cmap, norm=norm)
                ax_ref.set_title(f"Ref-{t}", fontsize=6)
                ax_ref.axis('off')
                ax_ref.text(2, 10, f"CSI: {refcsi:.2f}\nFAR: {reffar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f'saved in {self.save_dir}/epoch_{step}_batchidx_{batchidx}.png')

            
    def save_pixel_image_6methods(self, vis_args, step, batchidx, file_name="vis"):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            methods = ["tar", "pred", "ref", "far", "win", "lose"]
            csi_keys = ["", "csi_pred", "csi_ref", "csi_far", "csi_win", "csi_lose"]
            far_keys = ["", "far_pred", "far_ref", "far_far", "far_win", "far_lose"]

            # 转 numpy
            np_imgs = {}
            for m in methods:
                arr = vis_args[m].detach().cpu().numpy()  # b,t,c,h,w
                np_imgs[m] = arr[0, :, 0] * 255

            # 取等间隔6帧
            total_frames = np_imgs["tar"].shape[0]
            frame_ids = [0, 4, 8, 12, 16, 17]

            # fig, axes = plt.subplots(len(methods), len(frame_ids), figsize=(18, 12))  # 6x6
            # cmap, norm = vil_cmap()

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 12),
                constrained_layout=False  # 我们手动调节间距
            )
            cmap, norm = vil_cmap()

            # 减小行列间距
            plt.subplots_adjust(
                wspace=0.05,  # 列间距（越小越紧凑）
                hspace=0.05   # 行间距
            )


            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    ax.imshow(np_imgs[m][t], cmap=cmap, norm=norm)
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")

                    # 最后一列才写 CSI / FAR
                    if col == len(frame_ids) - 1 and csi_keys[row] != "":
                        csi_val = vis_args[csi_keys[row]]
                        far_val = vis_args[far_keys[row]]

                        # 在最后一张图右边空白区域写 CSI/FAR
                        ax.text(
                            1.05, 0.5, f"CSI: {csi_val:.2f}\nFAR: {far_val:.2f}",
                            transform=ax.transAxes,  # 轴坐标
                            fontsize=14, color="black", va="center", ha="left",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3")
                        )

                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")

    def save_pred_gt_images(self, pred, gt, step=0, batchidx=0, file_name="vis"):
        """
        可视化预测(pred)与真实(gt)的多帧对比。
        pred, gt: Tensor [b, t, 1, h, w]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            methods = ["gt", "pred"]
            # 转 numpy 并归一化到 0-255
            np_imgs = {}
            np_imgs["gt"] = gt.detach().cpu().numpy()[0, :, 0] * 255  # [t,h,w]
            np_imgs["pred"] = pred.detach().cpu().numpy()[0, :, 0] * 255

            total_frames = np_imgs["gt"].shape[0]
            frame_ids = [0, 4, 8, 12, 16, min(17, total_frames-1)]  # 等间隔6帧，避免越界

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 6),  # 两行6列
                constrained_layout=False
            )

            cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            # 调整行列间距
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    ax.imshow(np_imgs[m][t], cmap=cmap, norm=norm)
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")
                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")
            

    def save_input_images(self, input_image, step, batchidx, file_name="vis_input"):
        """
        绘制 input 的所有 6 帧图像并保存，带 colorbar
        input_image: Tensor [b, t, c, h, w]，t=6
        """
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            arr = input_image.detach().cpu().numpy()  # b,t,c,h,w
            np_imgs = arr[0, :, 0] * 255  # 取第一个 batch，第一个通道
            total_frames = np_imgs.shape[0]  # 应该是 6

            fig, axes = plt.subplots(1, total_frames, figsize=(12, 3))
            cmap, norm = vil_cmap()

            ims = []
            for col in range(total_frames):
                ax = axes[col]
                im = ax.imshow(np_imgs[col], cmap=cmap, norm=norm)
                ims.append(im)
                ax.set_title(f"t={col+1}", fontsize=10)  # 用 1-based 编号
                ax.axis("off")

            # 单独加一个 colorbar axes，避免重叠
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(ims[0], cax=cbar_ax)
            cbar.set_label("Pixel Value", fontsize=10)

            plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9)  # 留出右边位置
            out_file = os.path.join(
                self.save_dir, f"{file_name[0]}_input_epoch_{step}_batchidx_{batchidx}.png"
            )
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"saved in {out_file}")


   
    def save_pixel_image_rl(self, pred, win_image, lose_image, target_img, step, file_name=''):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            # 提取 win/lose/target/pred 的 numpy 格式图像 (B, T, C, H, W) → (T, H, W)
            win_imgs = win_image.detach().cpu().numpy()[0, :, 0] * 255
            lose_imgs = lose_image.detach().cpu().numpy()[0, :, 0] * 255
            target_imgs = target_img.detach().cpu().numpy()[0, :, 0] * 255
            pred_imgs = pred.detach().cpu().numpy()[0, :, 0] * 255

            val_max = 255
            val_min = 0

            for t in range(win_imgs.shape[0]):
                win_img = win_imgs[t]
                lose_img = lose_imgs[t]
                tgt_img = target_imgs[t]
                pred_img = pred_imgs[t]

                # Plot 四张图：win, lose, target, pred
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
                ax1.imshow(win_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title('win_sample')
                ax2.imshow(lose_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title('lose_sample')
                ax3.imshow(tgt_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax3.set_title('target_sample')
                im4 = ax4.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax4.set_title('pred_sample')

                # 四个图共用一个 colorbar
                cbar = plt.colorbar(im4, ax=[ax1, ax2, ax3, ax4])

                save_path = f'{self.save_dir}/pixel_quad_step{step}_time{t}_{file_name[0]}.png'
                print(f'Saving: {save_path}')

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.clf()

    def save_pixel_image_rl_2(self, pred, win_image, lose_image, target_img, target_ref, step, file_name=''):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            win_imgs = win_image.detach().cpu().numpy()[0, :, 0] * 255
            lose_imgs = lose_image.detach().cpu().numpy()[0, :, 0] * 255
            target_imgs = target_img.detach().cpu().numpy()[0, :, 0] * 255
            pred_imgs = pred.detach().cpu().numpy()[0, :, 0] * 255
            ref_imgs = target_ref.detach().cpu().numpy()[0, :, 0] * 255  # 新增参考模型预测

            val_max = 255
            val_min = 0

            for t in range(win_imgs.shape[0]):
                win_img = win_imgs[t]
                lose_img = lose_imgs[t]
                tgt_img = target_imgs[t]
                pred_img = pred_imgs[t]
                ref_img = ref_imgs[t]  # 第五张图

                # Plot 五张图
                fig, axes = plt.subplots(1, 5, figsize=(30, 6))  # 增加图像数量
                titles = ['win_sample', 'lose_sample', 'target_sample', 'pred_sample', 'ref_sample']
                images = [win_img, lose_img, tgt_img, pred_img, ref_img]

                for ax, img, title in zip(axes, images, titles):
                    im = ax.imshow(img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                    ax.set_title(title)
                    ax.axis('off')

                # 添加 colorbar
                cbar = plt.colorbar(im, ax=axes, orientation='vertical')

                save_path = f'{self.save_dir}/pixel_penta_step{step}_time{t}_{file_name[0]}.png'
                print(f'Saving: {save_path}')

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.clf()
            

    def save_pixel_image_rl_3(self, pred, win_image, lose_image, target_img, target_ref, step, file_name='', metrics_win=None, metrics_lose=None, metrics_pred=None, metrics_pred_ref=None):
            import matplotlib.patches as mpatches

            if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
                win_imgs = win_image.detach().cpu().numpy()[0, :, 0] * 255
                lose_imgs = lose_image.detach().cpu().numpy()[0, :, 0] * 255
                target_imgs = target_img.detach().cpu().numpy()[0, :, 0] * 255
                pred_imgs = pred.detach().cpu().numpy()[0, :, 0] * 255
                ref_imgs = target_ref.detach().cpu().numpy()[0, :, 0] * 255

                val_max = 255
                val_min = 0
                thresholds = [16, 74, 133, 160, 181, 219]

                for t in range(win_imgs.shape[0]):
                    win_img = win_imgs[t]
                    lose_img = lose_imgs[t]
                    tgt_img = target_imgs[t]
                    pred_img = pred_imgs[t]
                    ref_img = ref_imgs[t]

                    # === 第一张图：原图像 + FAR/CSI 指标 ===
                    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
                    images = [win_img, lose_img, tgt_img, pred_img, ref_img]
                    metric_texts = [
                        f"win_sample\nFAR: {metrics_win['avg']['far']:.4f}\nCSI: {metrics_win['avg']['csi']:.4f}" if metrics_win else "win_sample",
                        f"lose_sample\nFAR: {metrics_lose['avg']['far']:.4f}\nCSI: {metrics_lose['avg']['csi']:.4f}" if metrics_lose else "lose_sample",
                        "target_sample",
                        f"pred_sample\nFAR: {metrics_pred['avg']['far']:.4f}\nCSI: {metrics_pred['avg']['csi']:.4f}" if metrics_pred else "pred_sample",
                        f"ref_sample\nFAR: {metrics_pred_ref['avg']['far']:.4f}\nCSI: {metrics_pred_ref['avg']['csi']:.4f}" if metrics_pred_ref else "ref_sample"
                    ]
                    for ax, img, text in zip(axes, images, metric_texts):
                        im = ax.imshow(img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                        ax.set_title(text, fontsize=10)
                        ax.axis('off')
                    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical')
                    save_path = f'{self.save_dir}/pixel_penta_step{step}_time{t}_{file_name[0]}.png'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
                    plt.clf()

                    pred_dict = {
                        "win": win_img,
                        "lose": lose_img,
                        "pred": pred_img,
                        "ref": ref_img
                    }
                    fig, axes = plt.subplots(len(thresholds), len(pred_dict), figsize=(len(pred_dict)*5, len(thresholds)*5))
                    if len(thresholds) == 1:
                        axes = np.expand_dims(axes, axis=0)

                    metric_map = {
                        "win": metrics_win,
                        "lose": metrics_lose,
                        "pred": metrics_pred,
                        "ref": metrics_pred_ref
                    }

                    for row, threshold in enumerate(thresholds):
                        t_bin = (tgt_img >= threshold).astype(np.uint8)
                        for col, (key, pred_img) in enumerate(pred_dict.items()):
                            p_bin = (pred_img >= threshold).astype(np.uint8)

                            # 创建 RGB mask 图
                            hit = (t_bin == 1) & (p_bin == 1)
                            miss = (t_bin == 1) & (p_bin == 0)
                            fa = (t_bin == 0) & (p_bin == 1)
                            cor = (t_bin == 0) & (p_bin == 0)

                            overlay = np.zeros((t_bin.shape[0], t_bin.shape[1], 3), dtype=np.uint8)

                            overlay[cor] = [230, 230, 230]     
                            overlay[hit] = [102, 194, 165]     
                            overlay[miss] = [141, 160, 203]  
                            overlay[fa] = [252, 141, 98]    

                            # 获取当前图像对应的指标信息
                            metrics = metric_map.get(key)
                            

                            far_val = metrics[threshold]['far']
                            csi_val = metrics[threshold]['csi']
                            title_text = f"{key}_th{threshold}\nFAR: {far_val:.4f}  CSI: {csi_val:.4f}"

                            axes[row, col].imshow(overlay)
                            axes[row, col].set_title(title_text, fontsize=9)
                            axes[row, col].axis("off")

                    patches = [
                        mpatches.Patch(color=(102/255, 194/255, 165/255), label='Hit'),         
                        mpatches.Patch(color=(141/255, 160/255, 203/255), label='Miss'),          
                        mpatches.Patch(color=(252/255, 141/255, 98/255), label='False Alarm'),  
                        mpatches.Patch(color=(230/255, 230/255, 230/255), label='Correct Negative'),  
                    ]
                    
                    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=12)
                    plt.tight_layout(rect=[0, 0.05, 1, 1])

                    vis_save_path = f'{self.save_dir}/pixel_hits_step{step}_time{t}_{file_name[0]}.png'
                    plt.savefig(vis_save_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"Saved hit/miss/fa/cor visualization: {vis_save_path}")


    def save_pixel_image_rl_4(self, pred, target_img, target_ref, step, file_name='', metrics_pred=None, metrics_pred_ref=None):
        import matplotlib.patches as mpatches

        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            target_imgs = target_img.detach().cpu().numpy()[0, :, 0] * 255
            pred_imgs = pred.detach().cpu().numpy()[0, :, 0] * 255
            ref_imgs = target_ref.detach().cpu().numpy()[0, :, 0] * 255

            val_max = 255
            val_min = 0
            thresholds = [16, 74, 133, 160, 181, 219]

            for t in range(target_imgs.shape[0]):
                tgt_img = target_imgs[t]
                pred_img = pred_imgs[t]
                ref_img = ref_imgs[t]

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                images = [tgt_img, pred_img, ref_img]
                metric_texts = [
                    "target_sample",
                    f"pred_sample\nFAR: {metrics_pred['avg']['far']:.4f}\nCSI: {metrics_pred['avg']['csi']:.4f}" if metrics_pred else "pred_sample",
                    f"ref_sample\nFAR: {metrics_pred_ref['avg']['far']:.4f}\nCSI: {metrics_pred_ref['avg']['csi']:.4f}" if metrics_pred_ref else "ref_sample"
                ]
                for ax, img, text in zip(axes, images, metric_texts):
                    im = ax.imshow(img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                    ax.set_title(text, fontsize=10)
                    ax.axis('off')
                fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical')
                save_path = f'{self.save_dir}/test_pixel_triplet_step{step}_time{t}_{file_name[0]}.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
                plt.clf()
                
                pred_dict = {
                    "pred": pred_img,
                    "ref": ref_img
                }
                fig, axes = plt.subplots(len(thresholds), len(pred_dict), figsize=(len(pred_dict)*5, len(thresholds)*5))
                if len(thresholds) == 1:
                    axes = np.expand_dims(axes, axis=0)

                metric_map = {
                    "pred": metrics_pred,
                    "ref": metrics_pred_ref
                }

                fig, axes = plt.subplots(len(pred_dict), len(thresholds), figsize=(len(thresholds)*5, len(pred_dict)*5))
                if len(pred_dict) == 1:
                    axes = np.expand_dims(axes, axis=0)

                for row, (key, p_img) in enumerate(pred_dict.items()):
                    metrics = metric_map.get(key)
                    
                    for col, threshold in enumerate(thresholds):
                        t_bin = (tgt_img >= threshold).astype(np.uint8)
                        p_bin = (p_img >= threshold).astype(np.uint8)

                        # 创建 RGB mask 图
                        hit = (t_bin == 1) & (p_bin == 1)
                        miss = (t_bin == 1) & (p_bin == 0)
                        fa = (t_bin == 0) & (p_bin == 1)
                        cor = (t_bin == 0) & (p_bin == 0)

                        overlay = np.zeros((t_bin.shape[0], t_bin.shape[1], 3), dtype=np.uint8)
                        overlay[cor] = [230, 230, 230]   
                        overlay[hit] = [102, 194, 165]   
                        overlay[miss] = [141, 160, 203]   
                        overlay[fa] = [252, 141, 98]      

                        try:
                            far_val = metrics[threshold]['far']
                            csi_val = metrics[threshold]['csi']
                            title_text = f"th{threshold}\nFAR: {far_val:.4f}  CSI: {csi_val:.4f}"
                        except:
                            title_text = f"th{threshold}\n(no data)"

                        axes[row, col].imshow(overlay)
                        axes[row, col].set_title(f"{key}\n{title_text}", fontsize=9)
                        axes[row, col].axis("off")

                # 图例
                patches = [
                    mpatches.Patch(color=(102/255, 194/255, 165/255), label='Hit'),         
                    mpatches.Patch(color=(141/255, 160/255, 203/255), label='Miss'),          
                    mpatches.Patch(color=(252/255, 141/255, 98/255), label='False Alarm'),  
                    mpatches.Patch(color=(230/255, 230/255, 230/255), label='Correct Negative'),  
                ]

                fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=12)
                plt.tight_layout(rect=[0, 0.05, 1, 1])

                vis_save_path = f'{self.save_dir}/test_pixel_hits_step{step}_time{t}_{file_name[0]}.png'
                plt.savefig(vis_save_path, dpi=100, bbox_inches='tight')
                plt.close()
                print(f"Saved hit/miss/fa/cor visualization: {vis_save_path}")
            
    def save_npy(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:    
            pred_imgs = pred_image.detach().cpu().numpy()[:1]
            target_imgs = target_img.detach().cpu().numpy()[:1]

            np.save(f'{self.save_dir}/pred_step{step}.npy', pred_imgs)
            np.save(f'{self.save_dir}/gt_step{step}.npy', target_imgs)

    def save_npy_specific_dir(self, pred_image, target_img, step, dir):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:    
            pred_imgs = pred_image.detach().cpu().numpy()[:1]
            target_imgs = target_img.detach().cpu().numpy()[:1]
            np.save(f'{dir}/pred/pred_step{step}.npy', pred_imgs)
            np.save(f'{dir}/gt/gt_step{step}.npy', target_imgs)             
    
    def cmap_dict(self, s):
        return {'cmap': get_cmap(s, encoded=True)[0],
                'norm': get_cmap(s, encoded=True)[1],
                'vmin': get_cmap(s, encoded=True)[2],
                'vmax': get_cmap(s, encoded=True)[3]}
    
    def save_vil_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] * 255

            pred_imgs = pxl_pred_imgs 
            target_imgs = pxl_target_imgs

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                ax1.imshow(pred_img, **self.cmap_dict('vil'))
                ax1.set_title(f'pred_vil_step{step}_time{t}')
                im2 = ax2.imshow(target_img, **self.cmap_dict('vil'))
                ax2.set_title(f'target_vil')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/vil_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()
    
    def save_vil_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 255

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, **self.cmap_dict('vil'))
        ax1.set_title(f'pred_vil_step{step}_60min')
        im2 = ax2.imshow(last_target_img, **self.cmap_dict('vil'))
        ax2.set_title(f'target_vil_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/vil_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)
        

class hko7_visualizer(object):
    def __init__(self, exp_dir, sub_dir='hko7_train_vis'):
        self.exp_dir = exp_dir
        self.hko_zr_a = 58.53
        self.hko_zr_b = 1.56
        self.sub_dir = sub_dir
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f'{self.exp_dir}/{sub_dir}_{timestamp}'
        os.makedirs(self.save_dir, exist_ok=True)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0]* 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0]* 255

            for t in range(pxl_pred_imgs.shape[0]):
                pxl_pred_img = pxl_pred_imgs[t]
                pxl_target_img = pxl_target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 255
                val_min = 0
                ax1.imshow(pxl_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(pxl_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()

    def save_pixel_image_4(self, pred_image, target_img, ref, step, batchidx, predcsi, predfar, refcsi, reffar):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy()  # b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :, 0] * 255 # to pixel

            ref_imgs = ref.detach().cpu().numpy()
            pxl_ref_imgs = ref_imgs[0, :, 0] * 255 # to pixel

            fig, axes = plt.subplots(3, pxl_pred_imgs.shape[0], figsize=(36, 4))  # 3 rows (GT, Pred, Ref), T columns
            val_max = 255
            val_min = 0
            for t in range(pxl_pred_imgs.shape[0]):
                # GT
                ax_gt = axes[0, t]
                ax_gt.imshow(pxl_target_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_gt.set_title(f"GT-{t}", fontsize=6)
                ax_gt.axis('off')

                # Prediction
                ax_pred = axes[1, t]
                ax_pred.imshow(pxl_pred_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_pred.set_title(f"Pred-{t}", fontsize=6)
                ax_pred.axis('off')
                ax_pred.text(2, 10, f"CSI: {predcsi:.2f}\nFAR: {predfar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

                # Reference
                ax_ref = axes[2, t]
                ax_ref.imshow(pxl_ref_imgs[t], cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax_ref.set_title(f"Ref-{t}", fontsize=6)
                ax_ref.axis('off')
                ax_ref.text(2, 10, f"CSI: {refcsi:.2f}\nFAR: {reffar:.2f}", 
                            fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"epoch_{step}_batchidx_{batchidx}.png")
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f'saved in {self.save_dir}/epoch_{step}_batchidx_{batchidx}.png')
            
    def save_dbz_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0] #* 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0] #* 255

            pred_imgs = self._pixel_to_dBZ(pxl_pred_imgs) #self._pixel_to_rainfall(pxl_pred_imgs)
            target_imgs = self._pixel_to_dBZ(pxl_target_imgs) #self._pixel_to_rainfall(pxl_target_imgs)

            for t in range(pred_imgs.shape[0]):
                pred_img = pred_imgs[t]
                target_img = target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = np.max(target_img)
                val_min = np.min(target_img)
                ax1.imshow(pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_dbz_step{step}_time{t}')
                im2 = ax2.imshow(target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_dbz')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/dbz_step{step}_time{t}.png', dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()
    
    def save_hko7_last_image_and_npy(self, pred_image, target_img, step, ceph_prefix):
        assert get_world_size() == 1
        pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
        pxl_pred_imgs = pred_imgs[0, :, 0] * 255 # to pixel

        target_imgs = target_img.detach().cpu().numpy()
        pxl_target_imgs = target_imgs[0, :,  0] * 255

        ## save last frame ##
        last_pred_img = pxl_pred_imgs[-1]
        last_target_img = pxl_target_imgs[-1]

        val_max = 255
        val_min = 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(last_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
        ax1.set_title(f'pred_pxl_step{step}_60min')
        im2 = ax2.imshow(last_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
        ax2.set_title(f'target_pxl_60min')
        cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

        plt.savefig(f'{self.save_dir}/pxl_step{step}_60min.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.clf()

        ## save npy to ceph ##
        with io.BytesIO() as f:
            np.save(f, pxl_pred_imgs)
            f.seek(0)
            self.client.put(f'{ceph_prefix}/pred_step{step}.npy', f)


    def save_pixel_image_6methods(self, vis_args, step, batchidx, file_name="vis"):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            methods = ["tar", "pred", "ref", "far", "win", "lose"]
            csi_keys = ["", "csi_pred", "csi_ref", "csi_far", "csi_win", "csi_lose"]
            far_keys = ["", "far_pred", "far_ref", "far_far", "far_win", "far_lose"]

            # 转 numpy
            np_imgs = {}
            for m in methods:
                arr = vis_args[m].detach().cpu().numpy()  # b,t,c,h,w
                np_imgs[m] = arr[0, :, 0] * 70 -10

            # 取等间隔6帧
            total_frames = np_imgs["tar"].shape[0]
            frame_ids = [0, 3, 6, 9, 12, 14]

            # fig, axes = plt.subplots(len(methods), len(frame_ids), figsize=(18, 12))  # 6x6
            # cmap, norm = vil_cmap()

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 12),
                constrained_layout=False  # 我们手动调节间距
            )
            # cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            # 减小行列间距
            plt.subplots_adjust(
                wspace=0.05,  # 列间距（越小越紧凑）
                hspace=0.05   # 行间距
            )


            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    im = ax.imshow(np_imgs[m][t], cmap='jet', vmin=0, vmax=60)  # ✅ 保存 im
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")

                    if col == len(frame_ids) - 1 and csi_keys[row] != "":
                        csi_val = vis_args[csi_keys[row]]
                        far_val = vis_args[far_keys[row]]
                        ax.text(
                            1.05, 0.5, f"CSI: {csi_val:.2f}\nFAR: {far_val:.2f}",
                            transform=ax.transAxes,
                            fontsize=14, color="black", va="center", ha="left",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3")
                        )

                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

                # 给每行添加 colorbar，使用最后一帧的 im
            #     fig.colorbar(im, ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.04)
                
            # fig.colorbar(im, ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.04)
            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True) 
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")

    def save_input_images(self, input_image, step=0, batchidx=0, file_name="vis_input"):
        """
        绘制 input 的所有 6 帧图像并保存，带 colorbar
        input_image: Tensor [b, t, c, h, w]，t=6
        """
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            arr = input_image.detach().cpu().numpy()  # b,t,c,h,w
            np_imgs = arr[0, :, 0] * 70-10  # 取第一个 batch，第一个通道
            total_frames = np_imgs.shape[0]  # 应该是 6

            fig, axes = plt.subplots(1, total_frames, figsize=(12, 3))
            cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            ims = []
            for col in range(total_frames):
                ax = axes[col]
                im = ax.imshow(np_imgs[col], cmap='jet', vmin=0, vmax=60)
                ims.append(im)
                ax.set_title(f"t={col+1}", fontsize=10)  # 用 1-based 编号
                ax.axis("off")

            # 单独加一个 colorbar axes，避免重叠
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(ims[0], cax=cbar_ax)
            cbar.set_label("Pixel Value", fontsize=10)

            plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9)  # 留出右边位置
            out_file = os.path.join(
                self.save_dir, f"{file_name[0]}_input_epoch_{step}_batchidx_{batchidx}.png"
            )
            os.makedirs(os.path.dirname(out_file), exist_ok=True) 
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"saved in {out_file}")
            
    def save_pred_gt_images(self, pred, gt, step=0, batchidx=0, file_name="vis"):
        """
        可视化预测(pred)与真实(gt)的多帧对比。
        pred, gt: Tensor [b, t, 1, h, w]
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:
            methods = ["gt", "pred"]
            # 转 numpy 并归一化到 0-255
            np_imgs = {}
            np_imgs["gt"] = gt.detach().cpu().numpy()[0, :, 0] * 70-10  # [t,h,w]
            np_imgs["pred"] = pred.detach().cpu().numpy()[0, :, 0] * 70-10

            total_frames = np_imgs["gt"].shape[0]
            frame_ids = [0, 3, 6, 9, 12, 14]  # 等间隔6帧，避免越界

            fig, axes = plt.subplots(
                len(methods), len(frame_ids),
                figsize=(12, 6),  # 两行6列
                constrained_layout=False
            )

            # cmap, norm = cmap_dict()['cmap'], cmap_dict()['norm']

            # 调整行列间距
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            for row, m in enumerate(methods):
                for col, t in enumerate(frame_ids):
                    ax = axes[row, col]
                    im = ax.imshow(np_imgs[m][t], cmap='jet', vmin=0, vmax=60)  # 保存 im 对象
                    if row == 0:
                        ax.set_title(f"t={t}", fontsize=8)
                    ax.axis("off")

                axes[row, 0].set_ylabel(m.upper(), fontsize=8)

                # 给每行加 colorbar
                # fig.colorbar(im, ax=axes[row, :], orientation='vertical', fraction=0.02, pad=0.04)


            plt.tight_layout()
            out_file = os.path.join(self.save_dir, f"{file_name[0]}_epoch_{step}_batchidx_{batchidx}.png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)  # ✅ 确保目录存在
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"saved in {out_file}")
        
    def _pixel_to_rainfall(self, img, a=None, b=None):
        """Convert the pixel values to real rainfall intensity

        Parameters
        ----------
        img : np.ndarray
        a : float32, optional
        b : float32, optional

        Returns
        -------
        rainfall_intensity : np.ndarray
        """
        if a is None:
            a = self.hko_zr_a
        if b is None:
            b = self.hko_zr_b
        dBZ = self._pixel_to_dBZ(img)
        dBR = (dBZ - 10.0 * np.log10(a)) / b
        rainfall_intensity = np.power(10, dBR / 10.0)
        return rainfall_intensity
    
    def _pixel_to_dBZ(self, img):
        """

        Parameters
        ----------
        img : np.ndarray or float

        Returns
        -------

        """
        return img * 70.0 - 10.0


class shanghai_visualizer(object):
    def __init__(self, exp_dir, sub_dir='train_vis'):
        self.exp_dir = exp_dir
        self.sub_dir = sub_dir
        self.cmap_color = 'gist_ncar'
        self.save_dir = f'{self.exp_dir}/shanghai_{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_pixel_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0]* 255 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0]* 255

            for t in range(pxl_pred_imgs.shape[0]):
                pxl_pred_img = pxl_pred_imgs[t]
                pxl_target_img = pxl_target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 255
                val_min = 0
                ax1.imshow(pxl_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_pixel_step{step}_time{t}')
                im2 = ax2.imshow(pxl_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_pixel')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/pixel_step{step}_time{t}.png', dpi=150, bbox_inches='tight', pad_inches=0)
                plt.clf()


class SRAD2018_visualizer(object):
    def __init__(self, exp_dir, sub_dir='train_vis'):
        self.exp_dir = exp_dir
        self.sub_dir = sub_dir
        self.cmap_color = 'gist_ncar'
        self.save_dir = f'{self.exp_dir}/SRAD2018_{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_dbz_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0]* 80 # to pixel

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0]* 80

            for t in range(pxl_pred_imgs.shape[0]):
                pxl_pred_img = pxl_pred_imgs[t]
                pxl_target_img = pxl_target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 80
                val_min = 0
                ax1.imshow(pxl_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_dbz_step{step}_time{t}')
                im2 = ax2.imshow(pxl_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_dbz')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/dbz_step{step}_time{t}.png', dpi=150, bbox_inches='tight', pad_inches=0)
                plt.clf()


class NMIC_visualizer(object):
    def __init__(self, exp_dir, sub_dir='train_vis'):
        self.exp_dir = exp_dir
        self.sub_dir = sub_dir
        self.cmap_color = 'gist_ncar'
        self.save_dir = f'{self.exp_dir}/NMIC_{sub_dir}'
        os.makedirs(exist_ok=True, name=self.save_dir)

        self.client = Client("/mnt/shared-storage-user/xukaiyi/petrel-oss-python-sdk/petreloss.conf")
    
    def save_dbz_image(self, pred_image, target_img, step):
        if (get_world_size() > 1 and mpu.get_data_parallel_rank() == 0) or get_world_size() == 1:            
            pred_imgs = pred_image.detach().cpu().numpy() ##b, t, c, h, w
            pxl_pred_imgs = pred_imgs[0, :, 0]* 60 # to dbz

            target_imgs = target_img.detach().cpu().numpy()
            pxl_target_imgs = target_imgs[0, :,  0]* 60

            for t in range(pxl_pred_imgs.shape[0]):
                pxl_pred_img = pxl_pred_imgs[t]
                pxl_target_img = pxl_target_imgs[t]
                ## plot pred_img and target_img pair
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                val_max = 60
                val_min = 0
                ax1.imshow(pxl_pred_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax1.set_title(f'pred_dbz_step{step}_time{t}')
                im2 = ax2.imshow(pxl_target_img, cmap=self.cmap_color, vmin=val_min, vmax=val_max)
                ax2.set_title(f'target_dbz')
                cbar1 = plt.colorbar(im2, ax=[ax2, ax1])

                # mse = np.square(pred_img - target_img)
                # im3 = ax3.imshow(mse, cmap='OrRd')
                # ax3.set_title(f'mse')
                # cbar2 = plt.colorbar(im3, ax=ax3)
                plt.savefig(f'{self.save_dir}/dbz_step{step}_time{t}.png', dpi=150, bbox_inches='tight', pad_inches=0)
                plt.clf()


def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,  plot_stride=2,
        label_rotation=0,
        label_offset=(-0.06, 0.4),
        label_avg_int=False,
        fs=10,
        max_cols=10, ):
    """
    Parameters
    ----------
    seq:    Union[np.ndarray, Sequence[np.ndarray]]
        shape = (T, H, W). Float value 0-1 after `norm`.
    label:  Union[str, Sequence[str]]
        label for each sequence.
    norm:   Union[str, Dict[str, float]]
        seq_show = seq * norm['scale'] + norm['shift']
    interval_real_time: float
        The minutes of each plot interval
    max_cols: int
        The maximum number of columns in the figure.
    """

    def cmap_dict(s):
        return {'cmap': get_cmap(s, encoded=True)[0],
                'norm': get_cmap(s, encoded=True)[1],
                'vmin': get_cmap(s, encoded=True)[2],
                'vmax': get_cmap(s, encoded=True)[3]}

    # cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
    #                        'norm': get_cmap(s, encoded=True)[1],
    #                        'vmin': get_cmap(s, encoded=True)[2],
    #                        'vmax': get_cmap(s, encoded=True)[3]}

    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    # font.set_name('Times New Roman')
    fontproperties.set_size(fs)
    # font.set_weight("bold")

    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = label
    elif isinstance(seq, np.ndarray):
        seq_list = [seq.astype(np.float32), ]
        assert isinstance(label, str)
        label_list = [label, ]
    else:
        raise NotImplementedError
    if label_avg_int:
        label_list = [f"{ele1}\nAvgInt = {np.mean(ele2): .3f}"
                      for ele1, ele2 in zip(label_list, seq_list)]
    # plot_stride
    seq_list = [ele[::plot_stride, ...] for ele in seq_list]
    seq_len_list = [len(ele) for ele in seq_list]

    max_len = max(seq_len_list)

    max_len = min(max_len, max_cols)
    seq_list_wrap = []
    label_list_wrap = []
    seq_len_list_wrap = []
    for i, (seq, label, seq_len) in enumerate(zip(seq_list, label_list, seq_len_list)):
        num_row = math.ceil(seq_len / max_len)
        for j in range(num_row):
            slice_end = min(seq_len, (j + 1) * max_len)
            seq_list_wrap.append(seq[j * max_len: slice_end])
            if j == 0:
                label_list_wrap.append(label)
            else:
                label_list_wrap.append("")
            seq_len_list_wrap.append(min(seq_len - j * max_len, max_len))

    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
    nrows = len(seq_list_wrap)
    fig, ax = plt.subplots(nrows=nrows,
                           ncols=max_len,
                           figsize=(3 * max_len, 3 * nrows))

    for i, (seq, label, seq_len) in enumerate(zip(seq_list_wrap, label_list_wrap, seq_len_list_wrap)):
        ax[i][0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        for j in range(0, max_len):
            if j < seq_len:
                x = seq[j] * norm['scale'] + norm['shift']
                ax[i][j].imshow(x, **cmap_dict('vil'))
                if i == len(seq_list) - 1 and i > 0:  # the last row which is not the `in_seq`.
                    ax[-1][j].set_title(f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                                        y=-0.25, fontproperties=fontproperties)
            else:
                ax[i][j].axis('off')

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                       for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(-1.2, -0.),
                    borderaxespad=0, frameon=False, fontsize='10')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    print("start")
    import sys
    sys.path.append('/mnt/cache/gongjunchao/workdir/radar_forecasting')
    import torch
    from datasets.sevir import get_sevir_dataset
    dataset_kwargs = {
        'split': 'valid',
        'input_length': 13,
        'pred_length': 12,
        'data_dir': 'radar:s3://weather_radar_datasets/sevir'
    }
    # def __init__(self, split, input_length, pred_length, base_freq, height=480, width=480, **kwargs):
    dataset = get_sevir_dataset(**dataset_kwargs)
    # visualizer = hko7_visualizer(exp_dir='.')
    visualizer = sevir_visualizer(exp_dir='.')
    data_dict = dataset.__getitem__(3001)

    inp = data_dict['data_samples']
    visualizer.save_vil_image(inp.unsqueeze(0), inp.unsqueeze(0), 0)
    # vis_sevir_seq('test.png', inp.squeeze(1).numpy(), label='pred', interval_real_time=5.0, plot_stride=1, max_cols=11)
### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u visualizer.py ###
