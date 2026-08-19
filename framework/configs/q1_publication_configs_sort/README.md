# Q1 publication config pack

- q1_vector_gamma_serial: scan of gamma_bg for transition suppression / delay
- q1_vector_diagonal_serial_a: baseline diagonal in (eps, pump, noise, lowpass)
- q1_vector_diagonal_serial_b: shifted diagonal for robustness of phase boundary
- q1_ensemble_pair_spacing: two-pump spacing scan for pair interaction / ensemble behavior
- q1_ensemble_square4: four-core square arrangements for collective pattern dynamics

All configs use the same schema as your current sim_framework configs and route outputs into separate models/* directories.
