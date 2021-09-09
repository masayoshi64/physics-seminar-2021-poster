from argparse import ArgumentParser
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import sys

import qulacs
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import partial_trace
from qulacs.gate import DenseMatrix
import numpy as np
import numpy.linalg as npl
from scipy.linalg import expm
import matplotlib.pyplot as plt
import random
import os
from itertools import combinations
from tqdm import tqdm


class EntanglementType(Enum):
    from enum import auto

    TRIPARTITE = auto()
    ENTANGLEMENT_ENTROPY = auto()


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
        if dynamics == "lrc":
            self.add_LRC(k, n + k, depth)
        elif dynamics == "haar":
            self.circuit.add_random_unitary_gate(list(range(k, n + k)))
        elif dynamics == "heisenberg":
            self.add_Heisenberg(k, n + k, depth)
        elif dynamics == "fourbody":
            self.add_four_body_Heisenberg(k, n + k, depth)
        else:
            print("invalid dynamics type")

    def update(self):
        self.circuit.update_quantum_state(self.state)

    def add_LRC(self, l, r, depth):
        assert depth >= 0
        for d in range(depth):
            for i in range(l + d % 2, r - 1, 2):
                self.circuit.add_random_unitary_gate([i, i + 1])

    def add_Heisenberg(self, l, r, t):
        size = r - l
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        H = np.zeros((1 << size, 1 << size), dtype="complex128")
        for i in range(l, r - 1):
            Jx, Jy, Jz = self.get_coupling_constants()
            M = Jx * np.kron(X, X) + Jy * np.kron(Y, Y) + Jz * np.kron(Z, Z)
            if i > l:
                M = np.kron(np.identity(1 << (i - l)), M)
            if i + 2 < r:
                M = np.kron(M, np.identity(1 << (r - i - 2)))
            H += M
        U = DenseMatrix(list(range(l, r)), expm(-1j * t * H))
        self.circuit.add_gate(U)

    def add_four_body_Heisenberg(self, l, r, t):
        size = r - l
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        H = np.zeros((1 << size, 1 << size), dtype="complex128")
        for comb in combinations(range(l, r), 4):
            Jx, Jy, Jz = self.get_coupling_constants()
            Mx, My, Mz = np.ones(1), np.ones(1), np.ones(1)
            for i in range(l, r):
                if i in comb:
                    Mx = np.kron(Mx, X)
                    My = np.kron(My, Y)
                    Mz = np.kron(Mz, Z)
                else:
                    Mx = np.kron(Mx, np.identity(2))
                    My = np.kron(My, np.identity(2))
                    Mz = np.kron(Mz, np.identity(2))
            H += Mx * Jx + My * Jy + Mz * Jz
        U = DenseMatrix(list(range(l, r)), expm(-1j * t * H))
        self.circuit.add_gate(U)

    def get_coupling_constants(self):
        if len(self.cc) == 3:
            return self.cc
        return np.random.randn(3)

    # l1 norm
    def L1(self, rad_qubits):
        n, k = self.n, self.k
        l = len(rad_qubits)
        mat_size = pow(2, n + k - l)
        trace = partial_trace(self.state, rad_qubits)
        return npl.norm(trace.get_matrix() - np.identity(mat_size) / mat_size, "nuc")

    # mutual information
    def MI(self, rad_qubits):
        n, k = self.n, self.k
        l = len(rad_qubits)
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
def simulate(model, l_max, iter_num):
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
        for l in range(l_max + 1):
            data_L1[l][i] = model.L1(rad_qubits[:l])
            data_MI[l][i] = model.MI(rad_qubits[:l])
            data_CI[l][i] = model.CI(rad_qubits[:l])
        model.reset()
    return data_L1, data_MI, data_CI


def save_data(data, prefix):
    if not os.path.exists("data"):
        os.makedirs("data")
    np.savetxt("data/" + prefix + "_L1.csv", data[0], delimiter=",")
    np.savetxt("data/" + prefix + "_MI.csv", data[1], delimiter=",")
    np.savetxt("data/" + prefix + "_CI.csv", data[2], delimiter=",")


def main():
    parser = ArgumentParser()
    parser.add_argument("n", type=int, help="the number of qubits in blackhole")
    parser.add_argument("k", type=int, help="the number of alice's qubits")
    parser.add_argument("t", type=str)
    parser.add_argument("r", type=int)
    parser.add_argument("--depth", "-d", type=int, help="depth of circuit")
    parser.add_argument(
        "--coupling-constant", "-cc", type=str, help="space separated ints"
    )
    args = parser.parse_args()

    n, k, type, depth, r = args.n, args.k, args.t, args.depth, args.r
    cc = args.coupling_constant
    if cc is None:
        cc = []
    else:
        cc = list(map(float, cc.split()))
    blackhole = YoungBlackHole(n, k, type, depth, cc)
    output = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_n{n:02d}_k{k:02d}_type_{type}_r{r:03d}"
    if depth:
        output += f"_depth{depth:02d}"
    if cc:
        output += f"_cc{cc[0]:01d}_cc{cc[0]:01d}_{cc[1]:01d}_{cc[2]:01d}"
    save_data(simulate(blackhole, n, r), output)


if __name__ == "__main__":
    main()
