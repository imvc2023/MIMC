import os
import json

import matplotlib.pyplot as plt

def plot_loss(losses, losses_dir, datasetname):
    plt.cla()
    plt.xlabel("Epoch", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel("Total Loss", fontdict={'family': 'Times New Roman', 'size': 18})
    plt.plot(losses)
    plt.grid()
    figpath = losses_dir + datasetname + ".svg"
    plt.savefig(figpath)
    return figpath

if __name__ == "__main__":
    json_dir = "./losses/"
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            datasetname = file_name.split(".json")[0]
            with open(json_dir + file_name, 'r') as f:
                data = json.load(f)['data']
            plot_loss(data, "./re_loss/", datasetname)
