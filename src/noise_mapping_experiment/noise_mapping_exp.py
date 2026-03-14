from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit_ibm_runtime.fake_provider import (
    FakeManilaV2, FakeLimaV2, FakeQuitoV2, FakeBogotaV2, FakeRomeV2,
)
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit.transpiler import CouplingMap

# load in mock-IBM backend
def load_fake_backend(backend_name: str) -> FakeBackendV2:
    registry = {
        "FakeManilaV2": FakeManilaV2,
        "FakeLimaV2": FakeLimaV2,
        "FakeQuitoV2": FakeQuitoV2,
        "FakeBogotaV2": FakeBogotaV2,
        "FakeRomeV2": FakeRomeV2,
    }

    if backend_name not in registry:
        raise ValueError(f"Unknown backend '{backend_name}'")
    return registry[backend_name]()

# helper functions for errors
@dataclass
class BackendErrors:
    readout_error: List[float]  # len is num qubits
    twoq_error: Dict[Tuple[int, int], float]  # directed edge -> error rate
    coupling_edges: List[Tuple[int, int]]     # directed edges (approx)

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def get_backend_errors(backend: FakeBackendV2) -> BackendErrors:
    n = backend.num_qubits

    readout = [0.0 for _ in range(n)]
    twoq: Dict[Tuple[int, int], float] = {}

    coupling_map: CouplingMap = backend.coupling_map
    edges = list(coupling_map.get_edges())

    tgt = backend.target

    # measurement errors
    if "measure" in tgt:
        for q in range(n):
            try:
                props = tgt["measure"][(q,)]
                readout[q] = _safe_float(getattr(props, "error", None), default=0.0)
            except Exception:
                pass

    # two-qubit gate errors
    for inst_name in ("cx", "ecr"):
        if inst_name in tgt:
            for qargs, props in tgt[inst_name].items():
                if len(qargs) == 2:
                    twoq[(qargs[0], qargs[1])] = _safe_float(
                        getattr(props, "error", None), default=0.0
                    )
            break

    # ensure every coupling edge has a value
    for (u, v) in edges:
        twoq.setdefault((u, v), 0.02)

    return BackendErrors(
        readout_error=readout,
        twoq_error=twoq,
        coupling_edges=edges,
    )

# CIRCUITS
def bell_circuit() -> Tuple[QuantumCircuit, Set[str]]:
    # see https://quantum.cloud.ibm.com/docs/en/guides/hello-world
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)    # measure qubit 0 into classical bit 0
    qc.measure(1, 1)    # measure qubit 1 into classical bit 1
    # Qiskit returns counts keys with classical bits reversed: c1c0
    return qc, {"00", "11"}

def ghz_circuit(n: int = 3) -> Tuple[QuantumCircuit, Set[str]]:
    # see https://quantum.cloud.ibm.com/docs/en/guides/hello-world
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.measure(i, i)    # qubits corresponding with respective classical bits
    return qc, {"0" * n, "1" * n}

def int_to_qubit(qc: QuantumCircuit, qreg: list, value: int, nbits: int):
    # converts integer to qubit
    bits = format(value, f"0{nbits}b")[::-1] # reverse for little endian qiskit
    for i, bit in enumerate(bits):
        if bit == "1":
            qc.x(qreg[i])

def adder_canary(a: int = 1, b: int = 1, nbits: int = 2) -> Tuple[QuantumCircuit, Set[str]]:
    """
    uses CDKMRippleCarryAdder(kind="fixed") for addition on a and b
    - use helper qubit
    - measure 'b' register which holds sum
    - |a>|b> → |a>|a + b mod 2^n>
        - register b holds the sum after
    """
    # see https://medium.com/qiskit/a-guide-to-the-qiskit-circuit-library-36ee0f189956
    adder = CDKMRippleCarryAdder(nbits, kind="fixed")

    from qiskit.circuit import ClassicalRegister
    creg = ClassicalRegister(nbits, "meas_b")   # classical register to measure B
    qc = QuantumCircuit(*adder.qregs, creg)     # adder quantum bits, creg classical bits

    # name registers
    a_reg = next(r for r in qc.qregs if r.name == "a")
    b_reg = next(r for r in qc.qregs if r.name == "b")

    # input integers -> qubits
    int_to_qubit(qc, a_reg, a, nbits)
    int_to_qubit(qc, b_reg, b, nbits)

    qc.compose(adder, inplace=True)     # add

    for i in range(nbits):
        qc.measure(b_reg[i], creg[i])   # compare quantum bit i to classical i

    expected_sum = (a + b) % (2 ** nbits)
    bitstring = format(expected_sum, f"0{nbits}b")
    return qc, {bitstring}


def circuit_dict() -> Dict[str, Tuple[QuantumCircuit, Set[str]]]:
    return {
        "Bell": bell_circuit(),
        "GHZ-3": ghz_circuit(3),
        "Adder(2b): 1+1": adder_canary(a=1, b=1, nbits=2),
    }

# MAPPINGS
def all_possible_mappings(num_phys: int, num_logical: int):
    # returns all possible mappings of logical qubits to physical
    for perm in itertools.permutations(range(num_phys), num_logical):
        yield list(perm)

# MURALI-INSPIRED MAPPING
def best_path_reliability(
    n_qubits: int,
    edges: List[Tuple[int, int]],
    twoq_error: Dict[Tuple[int, int], float],
    ctrl: int,   # control (physical location)
    tar: int    # target (physical location)
) -> float:
    """
    Murali-inspired mapping, using Dijkstra's algorithm on -log(reliability) as weights.
    - Return max product-path reliability between control and target.
    - Reliability(edge) = 1 - error
    - Reliability(path) = product of edge reliabilities
    """
    if ctrl == tar:  # no search needed
        return 1.0

    # undirected adjacency graph for search
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_qubits)}
    for (u, v) in edges:
        # get reliability between edges u and v
        r_uv = 1.0 - _safe_float(twoq_error.get((u, v), 0.0), 0.0)
        r_vu = 1.0 - _safe_float(twoq_error.get((v, u), 0.0), 0.0)
        # use max reliability if edge reliabilities differ
        r = max(min(r_uv, 1.0), 1e-12)
        r = max(r, max(min(r_vu, 1.0), 1e-12))
        w = -math.log(r) # weight
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Dijkstra's algorithm
    import heapq    # priority queue module
    INF = 1e18
    dist = [INF] * n_qubits
    dist[ctrl] = 0.0
    pq = [(0.0, ctrl)]  # maps distance (cost) from source to vertex
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == tar:    # reached the end
            break
        for (v, w) in adj[u]:
            nd = d + w
            if nd < dist[v]:    # better path found (more reliable)
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    if dist[tar] >= INF / 2:
        return 0.0

    return float(math.exp(-dist[tar]))

def score_layout_murali_style(
    qc: QuantumCircuit,
    layout: List[int],
    be: BackendErrors,
    omega: float = 0.5,
) -> float:
    """
    Murali-inspired scoring:
      score = omega * sum log(readout_reliability(mapped qubits))
        + (1-omega) * sum log(best_path_reliability for each logical CNOT edge)
    """
    n_phys = len(be.readout_error)
    edges = be.coupling_edges
    twoq = be.twoq_error

    # readout term
    eps = 1e-12
    read_log = 0.0  # total readout reliability
    for _, pq in enumerate(layout):
        r = 1.0 - be.readout_error[pq]  # reliability score
        read_log += math.log(max(r, eps))

    # best path reliability for logical CNOT edges
    cnot_log = 0.0
    for inst, qargs, _ in qc.data:
        if inst.name == "cx":
            l_ctrl = qc.find_bit(qargs[0]).index
            l_tar = qc.find_bit(qargs[1]).index
            p_ctrl = layout[l_ctrl]
            p_tar = layout[l_tar]
            r_path = best_path_reliability(n_phys, edges, twoq, p_ctrl, p_tar) # best reliability
            cnot_log += math.log(max(r_path, eps))

    return omega * read_log + (1.0 - omega) * cnot_log


def pick_best_and_worst_by_calibration(
    qc: QuantumCircuit,
    be: BackendErrors,
    omega: float = 0.5,
) -> Tuple[List[int], List[int], float, float]:
    # returns best and worst layouts, based purely on calibration scores
    # calibrations scores computed with murali-style scoring
    n_phys = len(be.readout_error)
    n_log = qc.num_qubits
    layouts = list(all_possible_mappings(n_phys, n_log))
    scores = [score_layout_murali_style(qc, layout, be, omega=omega) for layout in layouts]
    best_i = int(np.argmax(scores))
    worst_i = int(np.argmin(scores))
    return layouts[best_i], layouts[worst_i], float(scores[best_i]), float(scores[worst_i])

def pick_tannu_vqa_like(qc: QuantumCircuit, be: BackendErrors) -> List[int]:
    """
    Tannu-inspired VQA: choose qubits based on highest connectivity strength
    - map in that order
    - connectivity strength = sum(edge success probability incident to q)
    """
    n_phys = len(be.readout_error)
    strength = [0.0 for _ in range(n_phys)]
    # compute connectivity strengths
    for (u, v) in be.coupling_edges:
        r = 1.0 - _safe_float(be.twoq_error.get((u, v), 0.0), 0.0) # 1 - gate error on edge
        r = max(min(r, 1.0), 0.0)
        strength[u] += r
        strength[v] += r
    phys_sorted = list(np.argsort(strength))[::-1]  # return strongest physical connectiosn
    return phys_sorted[: qc.num_qubits]

def pick_random_layout(qc: QuantumCircuit, num_phys: int, rng: random.Random) -> List[int]:
    # randomly map qubits
    return rng.sample(range(num_phys), qc.num_qubits)

# EXECUTION AND SUCCESS PROBS
def success_probability(counts: Dict[str, int], ok: Set[str]) -> float:
    # probability that circuit outputs an expected value
    total = sum(counts.values())
    good = sum(v for k, v in counts.items() if k in ok)     # outputs that match "ok" results
    return 0.0 if total == 0 else good / total  # successful shots / total shots

def run_once(
    qc: QuantumCircuit,
    backend,
    simulator: AerSimulator,
    layout: Optional[List[int]],
    shots: int,
) -> Tuple[float, Dict[str, int], Dict[str, int]]:
    from qiskit.result import Result
    tqc = transpile(qc, backend=backend, initial_layout=layout, optimization_level=0)   # maps the logical qubits onto physical device
    job = simulator.run(tqc, shots=shots)   # runs shots number of times
    res: Result = job.result()
    counts = res.get_counts()   # measurement histogram (dict)
    ops = dict(tqc.count_ops())     # how many of each gate in the circuit
    return float(tqc.depth()), counts, ops

def get_hardware_positions(be: BackendErrors, n_phys: int):
    """
    - gets 2D plotting coordinates from coupling graph (inferred)
    - linear if graph is a path (typical for fake manila)
    - fallback graph if not
    """
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(n_phys))

    for u, v in be.coupling_edges:
        G.add_edge(u, v)

    # graph = path?
    degrees = dict(G.degree())
    endpoints = [n for n, d in degrees.items() if d == 1]

    if len(endpoints) == 2 and all(d <= 2 for d in degrees.values()):   # path (1-2 connections per node)
        start = endpoints[0]
        order = list(nx.dfs_preorder_nodes(G, start))

        return {q: (i, 0.0) for i, q in enumerate(order)}

    # else generic graph
    pos = nx.spring_layout(G, seed=0)
    return {q: (float(pos[q][0]), float(pos[q][1])) for q in G.nodes()}

def plot_mapping_on_hardware(
    be: BackendErrors,
    layout: List[int],
    title: str,
    outpath: Optional[str] = None,
):
    # draw hardware mapping, highlight where logical qubits mapped onto physical
    # shows readout error per node

    n_phys = len(be.readout_error)  # num physical qubits
    pos = get_hardware_positions(be, n_phys)

    fig, ax = plt.subplots(figsize=(8, 5))

    # draw coupling edges (ensure no duplicates)
    drawn = set()
    for (u, v) in be.coupling_edges:
        e = tuple(sorted((u, v)))
        if e in drawn:
            continue
        drawn.add(e)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=3, alpha=0.6, zorder=1)

    used_phys = set(layout)

    # draw physical qubits
    for q in range(n_phys):
        x, y = pos[q]

        if q in used_phys:
            facecolor = "tab:blue"
        else:
            facecolor = "lightgray"

        ax.scatter(
            x, y,
            s=2200,
            c=facecolor,
            edgecolors="black",
            linewidths=2,
            zorder=2
        )

        # physical qubit layer
        ax.text(
            x, y - 0.35,
            f"phys {q}",
            ha="center", va="top",
            fontsize=11, color="black"
        )
        ax.text(
            x, y - 0.55,
            f"r_err={be.readout_error[q]:.3f}",
            ha="center", va="top",
            fontsize=9, color="black"
        )

    # logical qubit labels
    for lq, pq in enumerate(layout):
        x, y = pos[pq]
        ax.text(
            x, y + 0.02,
            f"L{lq}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white", zorder=3
        )

    # summary
    mapping_lines = [f"L{lq} → phys {pq}" for lq, pq in enumerate(layout)]
    mapping_text = "\n".join(mapping_lines)
    ax.text(
        0.02, 0.98,
        mapping_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    ax.set_title(title, fontsize=18)

    xs = [pos[q][0] for q in range(n_phys)]
    ys = [pos[q][1] for q in range(n_phys)]

    xpad = 0.5
    ypad = 0.8

    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout(pad=1.2)

    if outpath is not None:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

def save_mapping_visualizations(
    bench_name: str,
    be: BackendErrors,
    mappings: Dict[str, List[int]],
    out_dir: Optional[Path | str] = None
):
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "mapping_visualizations"
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_bench = bench_name.replace(" ", "_").replace(":", "").replace("/", "_")

    for method, layout in mappings.items():
        safe_method = method.replace(" ", "_")

        outpath = out_dir / f"{safe_bench}_{safe_method}_mapping.png"

        plot_mapping_on_hardware(
            be,
            layout,
            title=f"{bench_name} — {method}",
            outpath=str(outpath),
        )

        print(f"Saved mapping image: {outpath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None)    # directory for visualizations
    parser.add_argument("--backend", type=str, default="FakeManilaV2")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--omega", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    backend = load_fake_backend(args.backend)
    be = get_backend_errors(backend)

    # get noisy simulator from backend device
    simulator = AerSimulator.from_backend(backend)

    circuits = circuit_dict()

    # mapping strategies
    method_names = ["Random", "Best-by-calib", "Worst-by-calib", "Murali-inspired", "Tannu-VQA-like"]
    results: Dict[str, Dict[str, float]] = {name: {} for name in circuits.keys()}

    for bench_name, (qc, ok) in circuits.items():
        n_phys = len(be.readout_error)

        method_layouts: Dict[str, List[int]] = {}

        # random baseline (mean over trials)
        rand_scores = []
        rand_layout_example = None
        for t in range(args.random_trials):
            layout = pick_random_layout(qc, n_phys, rng)
            if t == 0:
                rand_layout_example = layout
            _, counts, _ = run_once(qc, backend, simulator, layout, args.shots)
            rand_scores.append(success_probability(counts, ok))
        results[bench_name]["Random"] = float(np.mean(rand_scores))
        method_layouts["Random"] = rand_layout_example

        # calibration mappings (best and worst)
        bestL, worstL, bestS, worstS = pick_best_and_worst_by_calibration(qc, be, omega=args.omega)
        _, best_counts, _ = run_once(qc, backend, simulator, bestL, args.shots)
        _, worst_counts, _ = run_once(qc, backend, simulator, worstL, args.shots)
        results[bench_name]["Best-by-calib"] = success_probability(best_counts, ok)
        results[bench_name]["Worst-by-calib"] = success_probability(worst_counts, ok)
        method_layouts["Best-by-calib"] = bestL
        method_layouts["Worst-by-calib"] = worstL

        # Murali-inspired
        results[bench_name]["Murali-inspired"] = results[bench_name]["Best-by-calib"]
        method_layouts["Murali-inspired"] = bestL

        # Tannu-inspired VQA-like
        tannuL = pick_tannu_vqa_like(qc, be)
        _, tannu_counts, _ = run_once(qc, backend, simulator, tannuL, args.shots)
        results[bench_name]["Tannu-VQA-like"] = success_probability(tannu_counts, ok)
        method_layouts["Tannu-VQA-like"] = tannuL

        print(f"\n[{bench_name}]")
        print(f"  Random (mean over {args.random_trials}): {results[bench_name]['Random']:.3f}")
        print(f"  Best-by-calib (score={bestS:.3f}):       {results[bench_name]['Best-by-calib']:.3f} layout={bestL}")
        print(f"  Worst-by-calib (score={worstS:.3f}):     {results[bench_name]['Worst-by-calib']:.3f} layout={worstL}")
        print(f"  Tannu-VQA-like:                          {results[bench_name]['Tannu-VQA-like']:.3f} layout={tannuL}")

        save_mapping_visualizations(bench_name, be, method_layouts, out_dir=args.out_dir)
    
    # plot success rates
    benches = list(results.keys())
    x = np.arange(len(benches))
    width = 0.16

    plt.figure()
    for i, m in enumerate(method_names):
        vals = [results[b][m] for b in benches]
        plt.bar(x + i * width, vals, width, label=m)

    plt.xticks(x + width * (len(method_names) - 1) / 2, benches, rotation=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Success probability (canary reward)")
    plt.title(f"Mapping sensitivity on {args.backend} (shots={args.shots})")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
