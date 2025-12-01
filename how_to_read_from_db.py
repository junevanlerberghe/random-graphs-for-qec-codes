from db_connector import GraphResultsDB


db = GraphResultsDB("database/qec_results.db")

er_runs = db.get_runs_by_model_p(model="ER", p=0.6, n_qubits=9, n_checks=4)
print(f"Found {len(er_runs)} ER runs with p=0.5, n_qubits=9, n_checks=4")
if er_runs:
    r0 = er_runs[0]
    print("First run:", r0["id"], r0["run_id"], r0["girth"], r0["edges"])

graph_runs_id, Q, C, B = db.get_biadjacency_by_model_p(
    model="WS",
    p=0.6,
    n_qubits=9,
    n_checks=4,
    index=0,   # first matching run
)
print("\WS example:")
print("graph_runs_id:", graph_runs_id)
print("Q labels:", Q)
print("C labels:", C)
print("B shape:", B.shape)

graph_runs_id_ws, pcm = db.get_parity_check_by_model_p(
    model="WS",
    p=0.6,
    index=0,
)
print("\nWS example:")
print("graph_runs_id:", graph_runs_id_ws)
print("PCM shape:", pcm.shape)


rows = db.get_code_metrics_by_model_p(
    model="ER",
    p=0.5,
)


for row in rows:
    print("\n--- code_metrics_id =", row["code_metrics_id"], " ---")
    print("graph_runs_id:                 ", row["graph_runs_id"])
    print("run_id:                        ", row["run_id"])
    print("n_qubits (metrics):           ", row["n_qubits"])
    print("k_logical:                    ", row["k_logical"])
    print("distance:                     ", row["distance"])
    print("code_rate:                    ", row["code_rate"])
    print("degenerate:             ", row["degenerate"])
    print("avg_logical_operator_weight:  ", row["avg_logical_operator_weight"])
    print("avg_stabilizer_weight:        ", row["avg_stabilizer_weight"])
    print("stabilizer_wep:         ", str(row["stabilizer_wep"]))
    print("normalizer_wep:         ", str(row["normalizer_wep"]))

db.close()


