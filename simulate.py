# coding: UTF-8
from argparse import ArgumentParser
from datetime import datetime
import json

from qulacs import QuantumState, QuantumCircuit
from qulacs.state import partial_trace
from qulacs.gate import DenseMatrix
import numpy as np
import numpy.linalg as npl
from scipy.linalg import expm
import pandas as pd
import random
import os
from itertools import combinations
from tqdm import tqdm

# [0, k): Charlie
# [k, 2k): Alice
# [2k, n+k): Black hole -> [k, n+k)


class YoungBlackHole:
    # coupling constant は指定しないとN(0, 1)に従う乱数
    def __init__(self, n, k, dynamics, depth=-1, cc=[]):
        self.n = n
        self.k = k
        self.size = n + k
        self.dynamics = dynamics
        self.depth = depth
        self.cc = cc
        self.reset()

    def reset(self):
        n, k = self.n, self.k
        dynamics = self.dynamics
        depth = self.depth
        self.state = QuantumState(n + k)
        self.circuit = QuantumCircuit(n + k)

        for i in range(k):
            self.circuit.add_H_gate(i)
            self.circuit.add_CNOT_gate(i, i + k)
        for i in range(n + k):
            self.circuit.add_random_unitary_gate([i])

        if dynamics == "lrc":
            self.add_LRC(k, n + k, depth)
        elif dynamics == "haar":
            self.circuit.add_random_unitary_gate(list(range(k, n + k)))
        elif dynamics == "heisenberg":
            self.add_Heisenberg(k, n + k, depth)
        elif dynamics == "twobody":
            self.add_all_to_all_Heisenberg(k, n + k, depth, 2)
        elif dynamics == "fourbody":
            self.add_all_to_all_Heisenberg(k, n + k, depth, 4)
        else:
            print("invalid dynamics type")

    def update(self):
        self.circuit.update_quantum_state(self.state)

    def add_LRC(self, left, right, depth):
        assert depth >= 0
        for d in range(depth):
            for i in range(left + d % 2, right - 1, 2):
                self.circuit.add_random_unitary_gate([i, i + 1])

    def add_Heisenberg(self, left, right, t):
        size = right - left
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        H = np.zeros((1 << size, 1 << size), dtype="complex128")
        for i in range(left, right - 1):
            Jx, Jy, Jz = self.get_coupling_constants()
            M = Jx * np.kron(X, X) + Jy * np.kron(Y, Y) + Jz * np.kron(Z, Z)
            if i > left:
                M = np.kron(np.identity(1 << (i - left)), M)
            if i + 2 < right:
                M = np.kron(M, np.identity(1 << (right - i - 2)))
            H += M
        U = DenseMatrix(list(range(left, right)), expm(-1j * t * H))
        self.circuit.add_gate(U)

    def add_all_to_all_Heisenberg(self, left, right, t, num):
        size = right - left
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        H = np.zeros((1 << size, 1 << size), dtype="complex128")
        for comb in combinations(range(left, right), num):
            Jx, Jy, Jz = self.get_coupling_constants()
            Mx, My, Mz = np.ones(1), np.ones(1), np.ones(1)
            for i in range(left, right):
                if i in comb:
                    Mx = np.kron(Mx, X)
                    My = np.kron(My, Y)
                    Mz = np.kron(Mz, Z)
                else:
                    Mx = np.kron(Mx, np.identity(2))
                    My = np.kron(My, np.identity(2))
                    Mz = np.kron(Mz, np.identity(2))
            H += Mx * Jx + My * Jy + Mz * Jz
        U = DenseMatrix(list(range(left, right)), expm(-1j * t * H))
        self.circuit.add_gate(U)

    def get_coupling_constants(self):
        if len(self.cc) == 3:
            return self.cc
        return np.random.randn(3)

    # l1 norm
    def L1(self, rad_qubits):
        n, k = self.n, self.k
        rad_num = len(rad_qubits)
        mat_size = pow(2, n + k - rad_num)
        trace = partial_trace(self.state, rad_qubits)
        return npl.norm(
            trace.get_matrix() -
            np.identity(mat_size) /
            mat_size,
            "nuc")

    # mutual information
    def MI(self, rad_qubits):
        n, k = self.n, self.k
        b_qubits = list(filter(lambda x: x not in rad_qubits, range(k, n + k)))
        AB = partial_trace(self.state, b_qubits)
        A = partial_trace(self.state, list(range(k, n + k)))
        B = partial_trace(self.state, list(range(k)) + b_qubits)
        return self.S(A) + self.S(B) - self.S(AB)

    # coherent information
    def CI(self, rad_qubits):
        n, k = self.n, self.k
        b_qubits = list(filter(lambda x: x not in rad_qubits, range(k, n + k)))
        AB = partial_trace(self.state, b_qubits)
        A = partial_trace(self.state, list(range(k, n + k)))
        return self.S(A) - self.S(AB)

    # entropy
    def S(self, rho):
        if rho is None:
            return 0
        mat = rho.get_matrix()
        eig_vals = npl.eigvalsh(mat)
        return -(eig_vals * np.log2(eig_vals + 0.000001)).sum()


# simulator for young black hole
def simulate(model, l_min, l_max, iter_num, prefix):
    n, k = model.n, model.k
    print("type:", model.dynamics)
    print(f"n={model.n}, k={model.k}")
    print(f"depth={model.depth}, coupling constant={model.cc}")
    data_MI = np.zeros((l_max + 1, iter_num))
    data_L1 = np.zeros((l_max + 1, iter_num))
    data_CI = np.zeros((l_max + 1, iter_num))
    for i in tqdm(range(iter_num)):
        rad_qubits = random.sample(list(range(k, n + k)), n)
        model.update()
        for rad_num in range(l_min, l_max + 1):
            data_L1[rad_num][i] = model.L1(rad_qubits[:rad_num])
            data_MI[rad_num][i] = model.MI(rad_qubits[:rad_num])
            data_CI[rad_num][i] = model.CI(rad_qubits[:rad_num])
        model.reset()
    df_L1 = pd.DataFrame(columns=["rad_num", "ave", "std"])
    df_MI = pd.DataFrame(columns=["rad_num", "ave", "std"])
    df_CI = pd.DataFrame(columns=["rad_num", "ave", "std"])
    for rad_num in range(l_min, l_max + 1):
        df_L1 = df_L1.append(
            pd.DataFrame(
                {
                    "rad_num": [rad_num],
                    "ave": [np.average(data_L1[rad_num])],
                    "std": [np.std(data_L1[rad_num])],
                }
            )
        )
        df_MI = df_MI.append(
            pd.DataFrame(
                {
                    "rad_num": [rad_num + 1],
                    "ave": [np.average(data_MI[rad_num])],
                    "std": [np.std(data_MI[rad_num])],
                }
            )
        )
        df_CI = df_CI.append(
            pd.DataFrame(
                {
                    "rad_num": [rad_num + 1],
                    "ave": [np.average(data_CI[rad_num])],
                    "std": [np.std(data_CI[rad_num])],
                }
            )
        )
    save_data(df_L1, prefix + "_L1.csv")
    save_data(df_MI, prefix + "_MI.csv")
    save_data(df_CI, prefix + "_CI.csv")


def save_data(df, file_name):
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/" + file_name, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("args", type=str)
    args = parser.parse_args()
    path = args.args

    with open(path) as f:
        df = json.load(f)
        n, k, type, depth, r = (
            df["n"],
            df["k"],
            df["type"],
            df["depth"],
            df["r"],
        )
    cc = df["coupling_constant"]

    blackhole = YoungBlackHole(n, k, type, depth, cc)
    prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    simulate(blackhole, 0, n, r, prefix)
    with open("data/" + prefix + ".json", mode="wt", encoding="utf-8") as f:
        json.dump(df, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
