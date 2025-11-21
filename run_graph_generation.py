import math
import os
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.generators import random_graph as bipartite_random_graph
from networkx.linalg.algebraicconnectivity import algebraic_connectivity, fiedler_vector
from networkx.algorithms.cycles import girth as nx_girth
import networkx as nx
import pandas as pd

from parity_check_from_adjacency import check_css_code_from_biadjacency, construct_css_parity_check_matrix, save_parity_check_matrix 

import networkx as nx
import random

def bipartite_watts_strogatz(n_qubits: int, n_checks: int, k_neighbors: int, p_flip: float):
    U = range(n_qubits)
    V = range(n_qubits, n_qubits+n_checks)
    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    # initial bipartite ring lattice
    for u in U:
        for i in range(k_neighbors):
            v = V[(u + i) % n_checks]
            G.add_edge(u, v)

    # rewiring
    for u, v in list(G.edges()):
        if random.random() < p_flip:
            # remove old edge
            G.remove_edge(u, v)
            # add new edge to random opposite-partition node
            if u in U:
                new_v = random.choice(list(V))
                G.add_edge(u, new_v)
            else:
                new_u = random.choice(list(U))
                G.add_edge(new_u, v)

    left = sorted([n for n, d in G.nodes(data=True) if d.get("bipartite", -1) == 0])
    right = sorted([n for n, d in G.nodes(data=True) if d.get("bipartite", -1) == 1])
    Q = [f"q{i}" for i in range(n_qubits)]
    C = [f"c{j}" for j in range(n_checks)]
    mapping = {left[i]: Q[i] for i in range(len(left))}
    mapping.update({right[j]: C[j] for j in range(len(right))})
    G = nx.relabel_nodes(G, mapping, copy=True)

    return G, Q, C


def make_bipartite_ER(n_qubits: int, n_checks: int, p: float, seed: int | None = None):
    G = bipartite_random_graph(n_qubits, n_checks, p, seed=seed)
    left = sorted([n for n, d in G.nodes(data=True) if d.get("bipartite", -1) == 0])
    right = sorted([n for n, d in G.nodes(data=True) if d.get("bipartite", -1) == 1])
    Q = [f"q{i}" for i in range(n_qubits)]
    C = [f"c{j}" for j in range(n_checks)]
    mapping = {left[i]: Q[i] for i in range(len(left))}
    mapping.update({right[j]: C[j] for j in range(len(right))})
    G = nx.relabel_nodes(G, mapping, copy=True)
    return G, Q, C


def biadjacency_matrix(G: nx.Graph, Q, C):
    B = np.zeros((len(Q), len(C)), dtype=int)
    for u, v in G.edges():
        if u in Q and v in C:
            B[Q.index(u), C.index(v)] = 1
        elif v in Q and u in C:
            B[Q.index(v), C.index(u)] = 1
    return B

def get_metrics_and_save_results(G, Q, C, OUT_DIR):
    # metrics
    degQ = [G.degree(q) for q in Q]
    degC = [G.degree(c) for c in C]
    avg_degQ = float(np.mean(degQ)) if len(degQ) else 0.0
    avg_degC = float(np.mean(degC)) if len(degC) else 0.0
    min_degQ, max_degQ = (int(np.min(degQ)), int(np.max(degQ))) if len(degQ) else (0,0)
    min_degC, max_degC = (int(np.min(degC)), int(np.max(degC))) if len(degC) else (0,0)

    # connectivity
    comps = list(nx.connected_components(G))
    num_components = len(comps)
    is_conn = (num_components == 1)

    # Algebraic connectivity (Fiedler value)
    try:
        lam2 = float(algebraic_connectivity(G))
    except Exception as e:
        lam2 = float("nan")

    # fiedler vector norm 
    try:
        fvec = np.asarray(fiedler_vector(G))
        fvec_norm = float(np.linalg.norm(fvec))
    except Exception:
        fvec_norm = float("nan")

    # girth
    g = nx_girth(G)
    if g == math.inf:
        g_out = float("inf")
    else:
        g_out = int(g)

    # edge stats
    E = G.number_of_edges()
    # bipartite density relative to |Q|*|C|
    density = E / (len(Q) * len(C)) if (len(Q) and len(C)) else 0.0

    # logical qubits (for your note)
    logical = n_qubits - n_checks*2

    # save artifacts
    # biadjacency
    B = biadjacency_matrix(G, Q, C)

    pcm = construct_css_parity_check_matrix(B)
    if pcm is None:
        #print("Invalid code: ", reason)
        return None

    print("Valid CSS code found!")

    tag = f"checks{n_checks}_p{p:.3f}_r{r:03d}"
    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    save_parity_check_matrix(pcm, os.path.join(out_dir, "parity_check_matrix.csv"))

    pd.DataFrame(B, index=Q, columns=C).to_csv(os.path.join(out_dir, "biadjacency.csv"))
    # edge list
    nx.write_edgelist(G, os.path.join(out_dir, "edges.csv"), data=False)

    # degrees table
    deg_df = pd.DataFrame({
        "node": Q + C,
        "partition": ["Q"]*len(Q) + ["C"]*len(C),
        "degree": degQ + degC,
    })
    deg_df.to_csv(os.path.join(out_dir, "degrees.csv"), index=False)

    # metrics row
    row = {
        "run_id": run_id,
        "n_qubits": n_qubits,
        "n_checks": n_checks,
        "p": p,
        "repeat": r,
        "edges": E,
        "density_QxC": density,
        "avg_degQ": avg_degQ,
        "min_degQ": min_degQ,
        "max_degQ": max_degQ,
        "avg_degC": avg_degC,
        "min_degC": min_degC,
        "max_degC": max_degC,
        "is_connected": bool(is_conn),
        "num_components": int(num_components),
        "fiedler_lambda2": lam2,
        "fiedler_vec_norm": fvec_norm,
        "girth": g_out,
        "logical_qubits": logical,
        "out_dir": out_dir
    }

    return row


if __name__ == "__main__":
    # Configuration
    n_qubits = 9
    checks_list = [4]
    p_list = [0.3, 0.4, 0.5, 0.6]

    # Repeats per (n_checks, p)
    repeats = 10000

    # Random seed base (each repeat uses seed_base + repeat_id)
    seed_base = 1234

    # Run sweep
    rows_ER = []
    rows_BA = []
    rows_WS = []
    run_id = 0

    for n_checks in checks_list:
        p_grid = list(p_list)

        for p in p_grid:
            count_valid_er = 0
            count_valid_ws = 0
            count_valid_ba = 0

            for r in range(repeats):
                run_id += 1
                seed = seed_base + r
                
                # generate ER
                G_ER, Q_ER, C_ER = make_bipartite_ER(n_qubits, n_checks, p, seed=seed)

                row = get_metrics_and_save_results(G_ER, Q_ER, C_ER, "outputs/results_ER")
                if row is not None:
                    count_valid_er += 1
                    rows_ER.append(row)

                G_WS, Q_WS, C_WS = bipartite_watts_strogatz(n_qubits, n_checks, k_neighbors=4, p_flip=p)
                row = get_metrics_and_save_results(G_WS, Q_WS, C_WS, "outputs/results_WS")
                if row is not None:
                    count_valid_ws += 1
                    rows_WS.append(row)
            
            print("Success rate of ER for p = ", p, " : ", count_valid_er / repeats)
            print("Success rate of WS for p = ", p, " and k=4 : ", count_valid_ws / repeats)

    metrics_path = os.path.join("outputs/results_ER", "graph_metrics.csv")
    metrics_df = pd.DataFrame(rows_ER)
    metrics_df.to_csv(metrics_path, index=False)

    metrics_path = os.path.join("outputs/results_WS", "graph_metrics.csv")
    metrics_df = pd.DataFrame(rows_WS)
    metrics_df.to_csv(metrics_path, index=False)