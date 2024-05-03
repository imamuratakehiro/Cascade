from utils.func import tsne_not_psd, file_exist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.manifold import TSNE

def tsne(songs):
    #dir_path = "./logfile/fig/tsne/transformer_bpm_bpm"
    dir_path = "./logfile/fig/tsne/transformer_bpm_bpm+chroma"
    path = {}
    # transformer+BPM, spec+bpm+hpsschroma, 5s, f2048_o512
    path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-04-20/csv/drums/normal_Test_e=0.csv"
    path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-10-09/csv/bass/normal_Test_e=0.csv"
    path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-12-45/csv/piano/normal_Test_e=0.csv"
    path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-15-15/csv/guitar/normal_Test_e=0.csv"
    path["residuals"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-29/18-17-45/csv/residuals/normal_Test_e=0.csv"
    # transformer+BPM
    #path["drums"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-40-07/csv/drums/normal.csv"
    #path["bass"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-42-03/csv/bass/normal.csv"
    #path["piano"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-43-36/csv/piano/normal.csv"
    #path["guitar"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-07/09-45-11/csv/guitar/normal.csv"
    #path["residuals"] = "/home/imamura23/nas02home/outputs/eval_nnet/runs/2024-01-10/10-07-33/csv/residuals/normal_Test_e=0.csv"
    inst_all = ["drums", "bass", "piano", "guitar", "residuals"]
    inst_all = ["drums", "bass", "piano", "guitar", "residuals"]
    label_vec = {}
    songs_all = {}
    for inst in inst_all:
        #print(f"\n{inst}")
        label_vec[inst] = pd.read_csv(path[inst], header=None).values
    #abel_mix = label_vec["drums"][:, 0:2]
    label_vec["mix"] = np.concatenate([
        label_vec["drums"],
        label_vec["bass"][:, 2:],
        label_vec["piano"][:, 2:],
        label_vec["guitar"][:, 2:],
        label_vec["residuals"][:, 2:]], axis=1)
    inst_all.append("mix")
    #cog = {}
    #recommend = {}
    color148 = list(matplotlib.colors.CSS4_COLORS.values())
    for inst in inst_all:
        #print(f"\n{inst}")
        #label_vec[inst] = pd.read_csv(path[inst], header=None).values
        songs_all[inst] = np.unique(label_vec[inst][:, 0])
        color10 = []
        vec10 = []
        """cog[inst] = {}
        for song in songs[inst]:
            #print(label_vec[inst][np.where(label_vec[inst][:, 0] == song)])
            vec_song = label_vec[inst][np.where(label_vec[inst][:, 0] == song)[0]][:, 2:]
            #print(vec_song.shape)
            cog[inst][song] = np.sum(vec_song, axis=0) / len(vec_song)"""
        #cog[inst] = cul_center_of_grabity(label_vec[inst], songs[inst])
        for n, song in enumerate(songs):
            #print(label_vec[inst][np.where(label_vec[inst][:, 0] == song)])
            #vec_song = label_vec[np.where(label_vec[:, 0] == song)[0]][:, 2:]
            #print(vec_song.shape)
            #cog[song] = np.sum(vec_song, axis=0) / len(vec_song)
            print(song)
            samesong_idx = np.where(label_vec[inst][:, 0] == float(song))[0]
            print(label_vec[inst][samesong_idx].shape)
            samesong_vec = label_vec[inst][samesong_idx][:, 2:]
            #samesong_label = songs_all[samesong_idx]
            #label20.append(label[samesong_idx])
            # 色を指定
            color10 = color10 + [n for _ in range(samesong_idx.shape[0])] # 色番号のみ格納
            #color10 = color10 + [color148[n] for _ in range(samesong_idx.shape[0])] # 136曲のとき使う
            # マークを指定
            """counter_m = -1 # 便宜上。本当は0にしたい
            log_ver = []
            for i in range(samesong_idx.shape[0]):
                if not samesong_label[i, 1] in log_ver:
                    log_ver.append(samesong_label[i, 1])
                    counter_m += 1
                marker10.append(markers[counter_m])"""
            vec10.append(samesong_vec)
        vec10 = np.concatenate(vec10, axis=0)
        perplexity = [5, 15, 30, 50]
        for i in range(len(perplexity)):
            fig, ax = plt.subplots(1, 1)
            X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i]).fit_transform(vec10)
            #for j in range(len(vec10)):
            #    mappable = ax.scatter(X_reduced[j, 0], X_reduced[j, 1], color=cm.tab20(color10[j]), s=30, cmap="tab20")
            #mappable = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color10, s=30) # 136曲の時使う
            mappable = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color10, s=30, cmap="tab20")
            #fig.colorbar(mappable, norm=BoundaryNorm(bounds,cmap.N))
            ax.legend(mappable.legend_elements(num=len(songs))[0], songs, borderaxespad=0, bbox_to_anchor=(1.05, 1),
                        loc="upper left", title="Songs")
            #ax.add_artist(legend1)
            file_exist(dir_path)
            fig.savefig(dir_path + f"/{inst}_p{i}_s{len(songs)}_v2.png", bbox_inches='tight')
            plt.clf()
            plt.close()
            print(inst, i)

def main():
    """songs = [
        1887,
        1916,
        1920,
        1930,
        1931,
        1940,
        1955,
        1961,
        1973,
        1980,
        1993,
        2000,
        2004,
        2010,
        2023,
        2044,
        2056,
        2062,
        2070,
        2081
    ]"""
    songs = [
        1887,
        1920,
        1930,
        1931,
        1940,
        1955,
        1961,
        1973,
        1980,
        1993,
        2000,
        2004,
        2010,
        2018,
        2023,
        2044,
        2056,
        2062,
        2081,
        2095
    ]
    """songs = [
    1876,
    1877,
    1878,
    1880,
    1881,
    1882,
    1883,
    1884,
    1886,
    1887,
    1888,
    1889,
    1891,
    1892,
    1893,
    1895,
    1896,
    1897,
    1898,
    1899,
    1900,
    1902,
    1903,
    1904,
    1907,
    1913,
    1916,
    1920,
    1925,
    1927,
    1928,
    1929,
    1930,
    1931,
    1932,
    1934,
    1935,
    1936,
    1937,
    1940,
    1943,
    1945,
    1947,
    1948,
    1949,
    1950,
    1951,
    1952,
    1954,
    1955,
    1956,
    1957,
    1959,
    1961,
    1962,
    1963,
    1965,
    1968,
    1972,
    1973,
    1974,
    1975,
    1976,
    1977,
    1978,
    1980,
    1981,
    1985,
    1986,
    1987,
    1989,
    1990,
    1993,
    1994,
    1995,
    1996,
    1998,
    2000,
    2001,
    2002,
    2003,
    2004,
    2005,
    2007,
    2008,
    2010,
    2013,
    2014,
    2016,
    2017,
    2018,
    2019,
    2023,
    2024,
    2026,
    2029,
    2030,
    2031,
    2032,
    2036,
    2037,
    2038,
    2040,
    2042,
    2044,
    2045,
    2046,
    2047,
    2048,
    2049,
    2050,
    2051,
    2052,
    2053,
    2054,
    2056,
    2061,
    2062,
    2063,
    2064,
    2069,
    2070,
    2074,
    2079,
    2081,
    2082,
    2084,
    2086,
    2087,
    2088,
    2090,
    2092,
    2093,
    2094,
    2095,
    2096
]"""
    tsne(songs)