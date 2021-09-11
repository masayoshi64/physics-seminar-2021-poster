import matplotlib.pyplot as plt
import pandas as pd
import os



def main():
    with open("labels.txt") as f:
        labels = f.readlines()
    with open("data_files.txt") as f:
        for i, name in enumerate(f.readlines()):
            if name[-1] == "\n":
                name = name[:-1]
            df = pd.read_csv("data/" + name)
            plt.errorbar(
                df["rad_num"], df["ave"], yerr=df["std"], capsize=5, label=labels[i]
            )
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    main()
