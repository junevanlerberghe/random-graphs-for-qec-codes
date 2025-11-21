import numpy as np
import networkx as nx

def gf2_rank(M):
    """Compute GF(2) rank using row reduction."""
    M = M.copy() % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def bipartite_components(B):
    """Return connected components of bipartite graph for biadjacency B."""
    r, n = B.shape
    G = nx.Graph()

    L_nodes = [f"L{i}" for i in range(r)]
    R_nodes = [f"R{j}" for j in range(n)]
    G.add_nodes_from(L_nodes, bipartite=0)
    G.add_nodes_from(R_nodes, bipartite=1)

    for i in range(r):
        for j in range(n):
            if B[i, j] == 1:
                G.add_edge(L_nodes[i], R_nodes[j])

    return list(nx.connected_components(G))


def check_css_code_from_biadjacency(B):
    """
    Check if B is a valid biadjacency matrix for a self-dual CSS code
    with H_X = H_Z = B.
    Returns (valid_code_bool, explanation_string).
    """
    if not np.all((B == 0) | (B == 1)):
        return False, "Matrix contains entries other than 0 or 1."

    r, n = B.shape

    if np.any((B @ B.T) % 2 != 0):
        return False, "Fails commutation check: B B^T != 0 mod 2."

    row_w = np.sum(B, axis=1)
    if np.any(row_w == 1):
        return False, "Contains a row of weight 1 (bad stabilizer)."

    col_w = np.sum(B, axis=0)
    if np.any(col_w == 0):
        return False, "Contains a column of weight 0 (qubit not checked)."

    components = bipartite_components(B)
    if len(components) > 1:
        return False, "Bipartite graph is disconnected."

    rank_B = gf2_rank(B)
    k = n - 2 * rank_B

    if k < 1:
        return False, f"Code encodes k={k} < 1 logical qubits."

    return True, f"Valid self-dual CSS code. k={k}, rank={rank_B}, n={n}."


def load_adjacency_matrix(file_path):
    """Load adjacency matrix from a given csv file path."""
    raw = np.genfromtxt(file_path, delimiter=",", dtype=str)
    data = raw[1:, 1:].astype(float)
    return data


def construct_css_parity_check_matrix(adj_matrix):
    """Construct CSS parity check matrix from adjacency matrix."""
    hx = adj_matrix.T
    hz = adj_matrix.T  # Assuming the same adjacency for Hz; modify if different

    valid, reason = check_css_code_from_biadjacency(hx)
    # print(f"Invalid CSS code: {reason}")
    if not valid:
        return None
    else:
        top = np.hstack((hx, np.zeros([len(hx), len(hz[0])])))
        bottom = np.hstack((np.zeros([len(hz), len(hx[0])]), hz))
        css_parity_check = np.vstack((top, bottom))
        return css_parity_check


def save_parity_check_matrix(parity_check_matrix, file_path):
    """Save parity check matrix to a given file path."""
    np.savetxt(file_path, parity_check_matrix, fmt='%d')


def find_logical_qubits(parity_check_matrix):
    """Calculate number of logical qubits from parity check matrix."""
    rank = gf2_rank(parity_check_matrix)
    n = parity_check_matrix.shape[1]
    k = n - rank
    return k