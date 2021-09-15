from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import json


def main():
    parser = ArgumentParser()
    parser.add_argument("args", type=str)
    args = parser.parse_args()
    path = args.args
    with open(path) as f:
        for i, name in enumerate(f.readlines()):
            if name[-1] == "\n":
                name = name[:-1]
            with open("data/" + name + ".json") as f:
                df = json.load(f)
                title = df["title"]

            df = pd.read_csv("data/" + name + "_L1.csv")
            # df = pd.read_csv("data/" + name + "_MI.csv")
            #df = pd.read_csv("data/" + name + "_CI.csv")

            plt.errorbar(
                df["rad_num"], df["ave"], yerr=df["std"], capsize=5, label=title
            )

    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    main()
