import numpy as np

class MarkovModel:
    """
    A simple Markov model for health economic evaluation.
    """
    def __init__(self, states, transition_matrix, discount_rate=0.03):
        self.states = states
        self.transition_matrix = transition_matrix
        self.discount_rate = discount_rate

    def run(self, cycles, initial_population, costs, qalys):
        """
        Runs the Markov model for a given number of cycles.
        """
        num_states = len(self.states)
        population_trace = np.zeros((cycles + 1, num_states))
        population_trace[0, :] = initial_population

        total_discounted_cost = 0
        total_discounted_qalys = 0

        for i in range(cycles):
            # Calculate population for the next cycle
            population_trace[i+1, :] = population_trace[i, :] @ self.transition_matrix

            # Calculate costs and QALYs for the current cycle
            cycle_cost = np.sum(population_trace[i, :] * costs)
            cycle_qalys = np.sum(population_trace[i, :] * qalys)

            # Apply discount rate
            discount_factor = (1 + self.discount_rate) ** i
            total_discounted_cost += cycle_cost / discount_factor
            total_discounted_qalys += cycle_qalys / discount_factor

        return total_discounted_cost, total_discounted_qalys

def run_cea(model_parameters, perspective='health_system', wtp_threshold=50000):
    """
    Runs a cost-effectiveness analysis for a given set of parameters and perspective.
    """
    # --- Unpack parameters ---
    states = model_parameters['states']
    tm_standard_care = np.array(model_parameters['transition_matrices']['standard_care'])
    tm_new_treatment = np.array(model_parameters['transition_matrices']['new_treatment'])
    cycles = model_parameters['cycles']
    initial_population = np.array(model_parameters['initial_population'])

    # --- Define costs and QALYs based on perspective ---
    if perspective == 'health_system':
        costs_standard = np.array(model_parameters['costs']['health_system']['standard_care'])
        costs_new = np.array(model_parameters['costs']['health_system']['new_treatment'])
        qalys_standard = np.array(model_parameters['qalys']['standard_care'])
        qalys_new = np.array(model_parameters['qalys']['new_treatment'])
    elif perspective == 'societal':
        # Health system costs
        hs_costs_standard = np.array(model_parameters['costs']['health_system']['standard_care'])
        hs_costs_new = np.array(model_parameters['costs']['health_system']['new_treatment'])
        # Societal costs (productivity, out-of-pocket, etc.)
        soc_costs_standard = np.array(model_parameters['costs']['societal']['standard_care'])
        soc_costs_new = np.array(model_parameters['costs']['societal']['new_treatment'])
        # Total societal costs
        costs_standard = hs_costs_standard + soc_costs_standard
        costs_new = hs_costs_new + soc_costs_new

        # For this simple model, we assume QALYs are the same from both perspectives
        qalys_standard = np.array(model_parameters['qalys']['standard_care'])
        qalys_new = np.array(model_parameters['qalys']['new_treatment'])
    else:
        raise ValueError("Perspective must be 'health_system' or 'societal'")

    # --- Run models ---
    model_sc = MarkovModel(states, tm_standard_care)
    cost_sc, qalys_sc = model_sc.run(cycles, initial_population, costs_standard, qalys_standard)

    model_nt = MarkovModel(states, tm_new_treatment)
    cost_nt, qalys_nt = model_nt.run(cycles, initial_population, costs_new, qalys_new)

    # --- Calculate ICER ---
    inc_cost = cost_nt - cost_sc
    inc_qalys = qalys_nt - qalys_sc

    if inc_qalys == 0:
        icer = np.inf
    else:
        icer = inc_cost / inc_qalys

    # --- Calculate Net Monetary Benefit (NMB) ---
    nmb_standard_care = (qalys_sc * wtp_threshold) - cost_sc
    nmb_new_treatment = (qalys_nt * wtp_threshold) - cost_nt
    incremental_nmb = nmb_new_treatment - nmb_standard_care

    return {
        "perspective": perspective,
        "cost_standard_care": cost_sc,
        "qalys_standard_care": qalys_sc,
        "cost_new_treatment": cost_nt,
        "qalys_new_treatment": qalys_nt,
        "incremental_cost": inc_cost,
        "incremental_qalys": inc_qalys,
        "icer": icer,
        "incremental_nmb": incremental_nmb
    }