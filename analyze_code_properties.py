
"""
Given a parity check matrix for a valid CSS code, perform the following:
1. run the WEP calculation (possibly brute force?)
2. from this, find the distance of the code
3. find average weight of stabilizers
4. find average weight of logical operators
5. calculate code rate (k/n)
"""

import os
import pandas as pd
import planqtn
import numpy as np
from planqtn.networks.stabilizer_measurement_state_prep import (
    StabilizerMeasurementStatePrepTN,
)

code_metrics = []
for code in os.listdir("outputs/results_ER"):
    if not os.path.isdir(f"outputs/results_ER/{code}"):
        continue
    pcm_file = f"outputs/results_ER/{code}/parity_check_matrix.csv"
    # get numpy array from pcm_file csv
    pcm = np.genfromtxt(pcm_file, delimiter=" ", dtype=int)
    print("Analyzing code: ", code)

    num_qubits = pcm.shape[1] // 2
    logical_qubits = num_qubits - pcm.shape[0]
    print("\t Number of qubits: ", num_qubits)
    print("\t Number of logical qubits: ", logical_qubits)
    print("\t Code rate (k/n): ", round(logical_qubits / num_qubits, 3))

    
    brute_force_node = StabilizerMeasurementStatePrepTN(pcm).conjoin_nodes()

    stabilizer_wep = (
        brute_force_node
        .stabilizer_enumerator_polynomial()
    )

    print("\t Stabilizer WEP: ", stabilizer_wep)

    normalizer_wep = stabilizer_wep.macwilliams_dual(num_qubits, logical_qubits, to_normalizer=True)
    print("\t Normalizer WEP: ", normalizer_wep)

    logical_wep = stabilizer_wep + normalizer_wep * -1
    non_zeros_dict = {k: v for k, v in logical_wep.items() if v != 0}
    distance = min(non_zeros_dict.keys())
    print("\t Distance: ", distance)

    logial_operator_density = 0
    for weight, count in logical_wep.items():
        logial_operator_density += weight * count
    logial_operator_density /= sum(logical_wep.dict.values())
    print("\t Avg logical operator weight: ", round(logial_operator_density, 3))

    stabilizer_density = 0
    for weight, count in stabilizer_wep.items():
        stabilizer_density += weight * count
    stabilizer_density /= sum(stabilizer_wep.dict.values())
    print("\t Avg stabilizer weight: ", round(stabilizer_density, 3))

    non_zeros_dict = {k: v for k, v in stabilizer_wep.items() if k != 0}
    min_weight = min(non_zeros_dict.keys())
    if(min_weight < distance):
        degenerate = True
    else:
        degenerate = False

    print("\t Degenerate: ", degenerate)

    row = {
        "code_id": code, 
        "n_qubits": num_qubits,
        "k_logical": logical_qubits,
        "distance": distance,
        "code_rate": round(logical_qubits / num_qubits, 3),
        "degenerate": degenerate,
        "avg_logical_operator_weight": round(logial_operator_density, 3),
        "avg_stabilizer_weight": round(stabilizer_density, 3),
        "stabilizer_wep": stabilizer_wep,
        "normalizer_wep": normalizer_wep,
    }

    code_metrics.append(row)

metrics_df = pd.DataFrame(code_metrics)
metrics_df.to_csv("outputs/results_ER/code_metrics.csv", index=False)