import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib
matplotlib.rcParams['animation.embed_limit']=200

class vals_anime():
    def __init__(self, vals, ax, title=None, xlabel=None, ylabel=None):
        self.ax = ax
        self.vals = vals
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            
        ax.set_xlim(0, len(vals))
        if len(ax.get_lines())==0:
            ax.set_ylim(min(vals), max(vals))
        else:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(min(vals),ymin), max(max(vals),ymax))
        self.ln0, = ax.plot(range(len(vals)), vals)
        self.ln, = ax.plot([], [], 'or', markersize=16)
        
    def init(self):
        return self.ln
        
    def update(self, i):
        self.ln.set_data(i, self.vals[i])
        return self.ln

class imgs_anime():
    def __init__(self, imgs, ax, title=None, time_norm=False, ticks=None):
        self.ax = ax
        if ticks is not None:
            ax.set_xticklabels(ticks, rotation=30)
            ax.set_yticklabels(ticks)

        self.imgs, vmin, vmax = self.normalize_imgs(imgs, time_norm)
        self.im = ax.imshow(self.imgs[0], vmin=vmin, vmax=vmax)
        if title is not None:
            ax.set_title(title)
        
    def init(self):
        return
    
    def update(self, i):
        self.im.set_data(self.imgs[i])
        return
    
    def normalize_imgs(self, imgs, time_norm=False):
        if time_norm:
            imgs = [img.abs() for img in imgs]
            vmin = min([img.min() for img in imgs])
            vmax = max([img.max() for img in imgs])
        else:
            imgs = [img.abs()/img.abs().max() for img in imgs]
            vmin = 0
            vmax = 1

        return imgs, vmin, vmax
    
class bars_anime():
    
    # data.shape = (frames, n_samples, n_types)
    #
    # Assuming that always first-order is positive and higher-order is negative.
    
    def __init__(self, data, ax, title=None, fix_range=False, ticks=None, ylabel=None):
        self.ax = ax
        if title is not None:
            ax.set_title(title)
        self.fix_range = fix_range
        self.data = data
        self.dmax = data.max()
        self.dmin = data.min()
        ax.set_xlim(-1, data.shape[1])
        ax.set_ylim(self.dmin, self.dmax)
        width = 0.8/data.shape[2]
        if ticks is not None:
            ax.set_xticklabels(ticks, rotation=30)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        
        x = np.arange(data.shape[1])
        self.rectss = []
        for i in range(data.shape[2]):
            if i==0:
                self.rectss.append(ax.bar(x+width*i-width*data.shape[2]/2, data[0,:,0], width=width, align="edge"))
            else:
                self.rectss.append(ax.bar(x+width*i-width*data.shape[2]/2, data[0,:,i], width=width, align="edge"))
        ax.legend(["delta","first","higher"])

    def init(self):
        return
        
    def update(self, i):
        self.set_rectss(self.rectss, self.data[i,:,:])
        if not self.fix_range:
            dmax_tmp, dmin_tmp = self.normalize_data(i)
            self.ax.set_ylim(dmin_tmp, dmax_tmp)
        return
        
    def set_rectss(self, rectss, d):
        for i, rects in enumerate(rectss):
            for j, rect in enumerate(rects):
                #rect.set_y(d[j,:i].sum())
                rect.set_height(d[j,i])
                
    def normalize_data(self, i):
        dmax_tmp = self.data[i,:,:].max()
        dmin_tmp = self.data[i,:,:].min()
        if abs(dmax_tmp/self.dmax) >= abs(dmin_tmp/self.dmin):
            dmin_tmp = dmax_tmp/self.dmax*self.dmin
        elif abs(dmax_tmp/self.dmax) < abs(dmin_tmp/self.dmin):
            dmax_tmp = dmin_tmp/self.dmin*self.dmax
        margin = (dmax_tmp-dmin_tmp)*0.1
        dmax_tmp += margin
        dmin_tmp -= margin
        
        return dmax_tmp, dmin_tmp

def vis_data(H, data, loss, acc):    
    fig, ax = plt.subplots(2, 2, figsize=(12,10))
    animes = []

    animes.append(imgs_anime(H, ax[0,0], "BH"))
    animes.append(bars_anime(data, ax[0,1], "LW contrib", True))
    ax[0,1].legend(["delta","fo","ho"])
    animes.append(vals_anime(loss, ax[1,0], "loss"))
    animes.append(vals_anime(acc, ax[1,1], "acc"))
        
    def init():
        for anime in animes:
            anime.init()
        fig.suptitle("i=0")
        return

    def update(i):
        #t = i*valfreq
        for anime in animes:
            anime.update(i)        
        fig.suptitle(f"i={i}")
        return

    ani = FuncAnimation(fig, update, init_func=init, frames=len(H),
                        interval=50)
    ani = ani.to_jshtml()
    #ani = ani.to_html5_video()
    plt.close()
    return ani#HTML(ani)
