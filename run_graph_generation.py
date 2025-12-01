import math
import numpy as np
from networkx.algorithms.bipartite.generators import random_graph as bipartite_random_graph
from networkx.linalg.algebraicconnectivity import algebraic_connectivity, fiedler_vector
from networkx.algorithms.cycles import girth as nx_girth
import networkx as nx

from parity_check_from_adjacency import check_css_code_from_biadjacency, construct_css_parity_check_matrix 

import networkx as nx
import random

from db_connector import GraphResultsDB


def bipartite_watts_strogatz(n_qubits: int, n_checks: int, k_neighbors: int, p_flip: float):
    U = range(n_qubits)
    V = range(n_qubits, n_qubits + n_checks)
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


def make_bipartite_BA(n_qubits: int, n_checks: int, m: int, seed: int | None = None):
    """
    Extended from networkx.generators.barabasi_albert.barabasi_albert_graph.
    """
    rng = np.random.default_rng(seed)

    U = list(range(n_qubits))
    V = list(range(n_qubits, n_qubits + n_checks))
    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)

    m0_u = m
    m0_v = m
    initial_U = U[:m0_u]
    initial_V = V[:m0_v]

    for u in initial_U:
        for v in initial_V:
            G.add_edge(u, v)

    # List of existing nodes per side, with nodes repeated once for each adjacent edge
    repeated_U = [u for u, d in G.degree(U) for _ in range(d)]
    repeated_V = [v for v, d in G.degree(V) for _ in range(d)]

    def _random_subset(seq, m, rng):
        targets = set()
        while len(targets) < m:
            x = rng.choice(seq)
            targets.add(x)
        return list(targets)

    for u in U[m0_u:]:
        targets = _random_subset(repeated_V, m, rng)
        for v in targets:
            G.add_edge(u, v)
            repeated_V.append(v)
            repeated_U.append(u)

    for v in V[m0_v:]:
        targets = _random_subset(repeated_U, m, rng)
        for u in targets:
            G.add_edge(u, v)
            repeated_U.append(u)
            repeated_V.append(v)

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


def get_metrics_and_save_results(G, Q, C, db: GraphResultsDB | None = None):
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

    # logical qubits 
    logical = n_qubits - n_checks*2

    # biadjacency
    B = biadjacency_matrix(G, Q, C)

    pcm = construct_css_parity_check_matrix(B)
    if pcm is None:
        return None

    print("Valid CSS code found!")

    tag = f"checks{n_checks}_p{p:.3f}_r{r:03d}"
    out_dir = None

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

    # return everything needed to store in DB
    return row, B, pcm, degQ, degC


if __name__ == "__main__":
    
    db = GraphResultsDB("database/qec_results.db")

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

    m_ba = 2

    for n_checks in checks_list:
        p_grid = list(p_list)

        for p in p_grid:
            count_valid_er = 0
            count_valid_ws = 0
            count_valid_ba = 0

            for r in range(repeats):
                run_id += 1
                seed = seed_base + r

                G_ER, Q_ER, C_ER = make_bipartite_ER(n_qubits, n_checks, p, seed=seed)
                result = get_metrics_and_save_results(G_ER, Q_ER, C_ER, db)
                if result is not None:
                    row, B, pcm, degQ, degC = result
                    row["model"] = "ER"
                    count_valid_er += 1
                    rows_ER.append(row)

                    graph_runs_id = db.insert_run(row, model="ER", m_ba=None, k_neighbors=None)

                    nodes = Q_ER + C_ER
                    partitions = ["Q"] * len(Q_ER) + ["C"] * len(C_ER)
                    degrees = degQ + degC

                    db.insert_biadjacency(graph_runs_id, row["run_id"], Q_ER, C_ER, B)
                    db.insert_edges(graph_runs_id, row["run_id"], G_ER.edges())
                    db.insert_degrees(graph_runs_id, row["run_id"], nodes, partitions, degrees)
                    db.insert_parity_check_matrix(graph_runs_id, row["run_id"], pcm)

                G_WS, Q_WS, C_WS = bipartite_watts_strogatz(
                    n_qubits, n_checks, k_neighbors=4, p_flip=p
                )
                result = get_metrics_and_save_results(G_WS, Q_WS, C_WS, db)
                if result is not None:
                    row, B, pcm, degQ, degC = result
                    row["model"] = "WS"
                    count_valid_ws += 1
                    rows_WS.append(row)

                    graph_runs_id = db.insert_run(row, model="WS", m_ba=None, k_neighbors=4)

                    nodes = Q_WS + C_WS
                    partitions = ["Q"] * len(Q_WS) + ["C"] * len(C_WS)
                    degrees = degQ + degC

                    db.insert_biadjacency(graph_runs_id, row["run_id"], Q_WS, C_WS, B)
                    db.insert_edges(graph_runs_id, row["run_id"], G_WS.edges())
                    db.insert_degrees(graph_runs_id, row["run_id"], nodes, partitions, degrees)
                    db.insert_parity_check_matrix(graph_runs_id, row["run_id"], pcm)

                G_BA, Q_BA, C_BA = make_bipartite_BA(n_qubits, n_checks, m_ba, seed=seed)
                result = get_metrics_and_save_results(G_BA, Q_BA, C_BA, db)
                if result is not None:
                    row, B, pcm, degQ, degC = result
                    row["model"] = "BA"
                    count_valid_ba += 1
                    rows_BA.append(row)

                    graph_runs_id = db.insert_run(row, model="BA", m_ba=m_ba, k_neighbors=None)

                    nodes = Q_BA + C_BA
                    partitions = ["Q"] * len(Q_BA) + ["C"] * len(C_BA)
                    degrees = degQ + degC

                    db.insert_biadjacency(graph_runs_id, row["run_id"], Q_BA, C_BA, B)
                    db.insert_edges(graph_runs_id, row["run_id"], G_BA.edges())
                    db.insert_degrees(graph_runs_id, row["run_id"], nodes, partitions, degrees)
                    db.insert_parity_check_matrix(graph_runs_id, row["run_id"], pcm)

            print("Success rate of ER for p = ", p, " : ", count_valid_er / repeats)
            print("Success rate of WS for p = ", p, " and k=4 : ", count_valid_ws / repeats)
            print("Success rate of BA for p = ", p, " and m=", m_ba, " : ", count_valid_ba / repeats)


            # save success rates into the DB
            db.insert_success_rate(
                model="ER",
                n_qubits=n_qubits,
                n_checks=n_checks,
                p=p,
                m_ba=None,
                k_neighbors=None,
                repeats=repeats,
                num_valid=count_valid_er,
            )
            db.insert_success_rate(
                model="WS",
                n_qubits=n_qubits,
                n_checks=n_checks,
                p=p,
                m_ba=None,
                k_neighbors=4,
                repeats=repeats,
                num_valid=count_valid_ws,
            )
            db.insert_success_rate(
                model="BA",
                n_qubits=n_qubits,
                n_checks=n_checks,
                p=p,
                m_ba=m_ba,
                k_neighbors=None,
                repeats=repeats,
                num_valid=count_valid_ba,
            )

    db.close()
