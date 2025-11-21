Three python files:
1. parity_check_from_adjacency.py - This contains functions to convert the graphs we generate to the parity check matrix of a quantum code. Also has functions for checking if it is a valid code.

2. run_graph_generation.py - Run this file to generate random codes. It will run ER and WS random graph models and save the results if a valid code is produced. 

3. analyze_code_properties.py - Given the parity check matrices generated from the above file, will read them and gather the properties of the resulting code for further analysis. 
