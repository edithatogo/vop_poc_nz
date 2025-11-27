# Technical Specification: Health Economic Analysis Model

## 1. Model Structure
The analysis uses a **Markov Model** to simulate the progression of a cohort through defined health states over time.

### Formulae

#### Markov Trace
The population distribution at cycle $t+1$ is calculated as:
$$ P_{t+1} = P_t \times T $$
Where:
*   $P_t$ is the row vector of state probabilities at cycle $t$.
*   $T$ is the transition probability matrix.

#### Costs and QALYs
Total discounted costs ($C$) and QALYs ($E$) are calculated as:
$$ C = \sum_{t=0}^{T} \frac{\sum_{s} (P_{t,s} \times c_s)}{(1+r)^t} $$
$$ E = \sum_{t=0}^{T} \frac{\sum_{s} (P_{t,s} \times u_s)}{(1+r)^t} $$
Where:
*   $P_{t,s}$ is the proportion of the cohort in state $s$ at cycle $t$.
*   $c_s$ is the cost associated with state $s$.
*   $u_s$ is the utility (QALY weight) associated with state $s$.
*   $r$ is the discount rate.

#### Incremental Cost-Effectiveness Ratio (ICER)
$$ ICER = \frac{C_{new} - C_{standard}}{E_{new} - E_{standard}} = \frac{\Delta C}{\Delta E} $$

#### Net Monetary Benefit (NMB)
$$ NMB = (E \times \lambda) - C $$
$$ \text{Incremental NMB} = (\Delta E \times \lambda) - \Delta C $$
Where $\lambda$ is the willingness-to-pay threshold (e.g., $50,000/QALY).

## 2. Assumptions
*   **Discount Rate:** 3% per annum (standard for NZ health economic evaluation).
*   **Time Horizon:** Varies by intervention (10 years to lifetime), specified in `parameters.yaml`.
*   **Perspective:**
    *   **Health System:** Includes direct medical costs.
    *   **Societal:** Includes health system costs + productivity losses + other societal costs/savings.
*   **Productivity Costs:** Calculated using either Human Capital or Friction Cost methods (configurable).

## 3. Distributive CEA (Equity)
Equity weights are applied to QALYs based on the socioeconomic status (SES) of the subgroup.
$$ \text{Equity-Weighted QALYs} = \sum_{g} (E_g \times w_g) $$

Where:
*   $E_g$: The total QALYs gained by subgroup $g$ (e.g., High SES, Low SES).
*   $w_g$: The equity weight assigned to subgroup $g$. In this model, we use:
    *   **Low SES / Māori:** $w = 1.5$ (Illustrative weight reflecting a pro-equity policy preference, consistent with Ministry of Health equity goals. Note: PHARMAC typically uses unweighted QALYs in base analysis).
    *   **High SES / Non-Māori:** $w = 1.0$ (baseline weight).

### Intersectionality Note
The current model applies equity weights to discrete, pre-defined subgroups (e.g., "Māori", "Low SES"). It does not automatically model the **intersectionality** of these groups (e.g., compounding weights for an individual who is both Māori *and* Low SES) unless such an intersectional subgroup is explicitly defined in the input data. Future iterations could incorporate multiplicative weighting for intersectional identities.
## 4. Value of Perspective (Decision Discordance)
The "Value of Perspective" quantifies the societal loss incurred when decisions are made based solely on the health system perspective.

$$ VoP = NMB_{societal}(d^*_{societal}) - NMB_{societal}(d^*_{health\_system}) $$

Where:
*   $d^*_{societal}$ is the optimal decision (highest NMB) from the societal perspective.
*   $d^*_{health\_system}$ is the optimal decision from the health system perspective.
*   $NMB_{societal}(d)$ is the Net Monetary Benefit of decision $d$ calculated using societal costs and benefits.

## 5. Value of Information (VOI)
VOI analysis quantifies the expected value of reducing uncertainty.

### Expected Value of Perfect Information (EVPI)
$$ EVPI = E_\theta [\max_d NMB(d, \theta)] - \max_d E_\theta [NMB(d, \theta)] $$
Where $\theta$ represents the vector of all uncertain parameters.

### Expected Value of Partial Perfect Information (EVPPI)
For a subset of parameters $\phi \subset \theta$:
$$ EVPPI = E_\phi [\max_d E_{\psi|\phi} [NMB(d, \phi, \psi)]] - \max_d E_\theta [NMB(d, \theta)] $$
Where $\psi = \theta \setminus \phi$ (the remaining uncertain parameters).
