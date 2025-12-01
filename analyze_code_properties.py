"""
Given a parity check matrix for a valid CSS code, perform the following:
1. run the WEP calculation (possibly brute force?)
2. from this, find the distance of the code
3. find average weight of stabilizers
4. find average weight of logical operators
5. calculate code rate (k/n)
"""

import planqtn
import numpy as np
from planqtn.networks.stabilizer_measurement_state_prep import (
    StabilizerMeasurementStatePrepTN,
)
from db_connector import GraphResultsDB


def main():
    db = GraphResultsDB("database/qec_results.db")

    graph_runs_ids = db.get_graph_runs_ids_with_pcm()
    print("Found", len(graph_runs_ids), "runs with parity-check matrices")

    for graph_runs_id in graph_runs_ids:
        run_info = db.get_run_by_id(graph_runs_id)
        if run_info is None:
            print("Skipping graph_runs_id =", graph_runs_id, "(no run_info)")
            continue

        run_id = run_info["run_id"]
        model = run_info["model"]
        p = run_info["p"]
        n_qubits_run = run_info["n_qubits"]
        n_checks_run = run_info["n_checks"]

        print(
            f"\nAnalyzing run: graph_runs_id={graph_runs_id}, "
            f"run_id={run_id}, model={model}, p={p}, "
            f"n_qubits={n_qubits_run}, n_checks={n_checks_run}"
        )

        pcm = db.get_parity_check_matrix_for_run(graph_runs_id)

        num_qubits = pcm.shape[1] // 2
        logical_qubits = num_qubits - pcm.shape[0]
        code_rate = float(logical_qubits) / float(num_qubits) if num_qubits > 0 else 0.0

        print("\tNumber of qubits: ", num_qubits)
        print("\tNumber of logical qubits: ", logical_qubits)
        print("\tCode rate (k/n): ", round(code_rate, 3))

        brute_force_node = StabilizerMeasurementStatePrepTN(pcm).conjoin_nodes()

        stabilizer_wep = brute_force_node.stabilizer_enumerator_polynomial()
        print("\tStabilizer WEP: ", stabilizer_wep)

        normalizer_wep = stabilizer_wep.macwilliams_dual(
            num_qubits, logical_qubits, to_normalizer=True
        )
        print("\tNormalizer WEP: ", normalizer_wep)

        logical_wep = stabilizer_wep + normalizer_wep * -1

        non_zeros_dict = {k: v for k, v in logical_wep.items() if v != 0}
        distance = min(non_zeros_dict.keys()) if non_zeros_dict else 0
        print("\tDistance: ", distance)

        logial_operator_density = 0.0
        total_logical = sum(logical_wep.dict.values())
        if total_logical > 0:
            for weight, count in logical_wep.items():
                logial_operator_density += weight * count
            logial_operator_density /= total_logical
        print(
            "\tAvg logical operator weight: ",
            round(logial_operator_density, 3),
        )

        stabilizer_density = 0.0
        total_stab = sum(stabilizer_wep.dict.values())
        if total_stab > 0:
            for weight, count in stabilizer_wep.items():
                stabilizer_density += weight * count
            stabilizer_density /= total_stab
        print(
            "\tAvg stabilizer weight: ",
            round(stabilizer_density, 3),
        )

        non_zeros_stab = {k: v for k, v in stabilizer_wep.items() if k != 0}
        min_weight = min(non_zeros_stab.keys()) if non_zeros_stab else 0
        if min_weight < distance:
            degenerate = True
        else:
            degenerate = False

        print("\tDegenerate: ", degenerate)

        db.insert_code_metrics(
            graph_runs_id=graph_runs_id,
            run_id=run_id,
            n_qubits=num_qubits,
            k_logical=logical_qubits,
            distance=distance,
            code_rate=code_rate,
            degenerate=degenerate,
            avg_logical_operator_weight=logial_operator_density,
            avg_stabilizer_weight=stabilizer_density,
            stabilizer_wep=stabilizer_wep,
            normalizer_wep=normalizer_wep,
        )

    db.close()


if __name__ == "__main__":
    main()
