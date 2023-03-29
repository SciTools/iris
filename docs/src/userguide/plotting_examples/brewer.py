import matplotlib.pyplot as plt
import numpy as np

import iris.palette


def main():
    a = np.linspace(0, 1, 256).reshape(1, -1)
    a = np.vstack((a, a))

    maps = sorted(iris.palette.CMAP_BREWER)
    nmaps = len(maps)

    fig = plt.figure(figsize=(7, 10))
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
    for i, m in enumerate(maps):
        ax = plt.subplot(nmaps, 1, i + 1)
        plt.axis("off")
        plt.imshow(a, aspect="auto", cmap=plt.get_cmap(m), origin="lower")
        pos = list(ax.get_position().bounds)
        fig.text(
            pos[0] - 0.01, pos[1], m, fontsize=8, horizontalalignment="right"
        )

    plt.show()


if __name__ == "__main__":
    main()
