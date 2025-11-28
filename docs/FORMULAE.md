# Mathematical Formulae

This document details the mathematical formulae implemented in the `vop_poc_nz` codebase for health economic evaluation.

## 1. Cost-Effectiveness Analysis (CEA)

### Incremental Cost-Effectiveness Ratio (ICER)

The ICER represents the additional cost per unit of health outcome gained.

$$
ICER = \frac{\Delta C}{\Delta E} = \frac{C_{new} - C_{comparator}}{E_{new} - E_{comparator}}
$$

Where:
- $C_{new}, C_{comparator}$ are the total discounted costs of the new intervention and comparator.
- $E_{new}, E_{comparator}$ are the total discounted health outcomes (QALYs).

### Net Monetary Benefit (NMB)

The NMB represents the value of an intervention in monetary terms at a given willingness-to-pay (WTP) threshold ($\lambda$).

$$
NMB = (E \times \lambda) - C
$$

### Incremental Net Monetary Benefit (iNMB)

$$
iNMB = NMB_{new} - NMB_{comparator} = (\Delta E \times \lambda) - \Delta C
$$

Decision Rule: If $iNMB > 0$, the new intervention is cost-effective at threshold $\lambda$.

## 2. Distributional Cost-Effectiveness Analysis (DCEA)

### Gini Coefficient

The Gini coefficient measures inequality in the distribution of net health benefits (NHB).

$$
G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i}
$$

Where:
- $x_i$ is the NHB of the $i$-th individual (or subgroup), sorted in non-decreasing order.
- $n$ is the number of individuals/subgroups.

### Atkinson Index

The Atkinson index measures inequality with an explicit aversion parameter ($\epsilon$).

$$
A_\epsilon = 1 - \frac{1}{\mu} \left( \frac{1}{n} \sum_{i=1}^{n} x_i^{1-\epsilon} \right)^{\frac{1}{1-\epsilon}}
$$

Where:
- $\epsilon$ is the inequality aversion parameter ($\epsilon \ge 0$).
- $\mu$ is the mean NHB.
- $x_i$ is the NHB of subgroup $i$.

For $\epsilon = 1$:
$$
A_1 = 1 - \frac{\prod_{i=1}^{n} x_i^{1/n}}{\mu}
$$

### Equally Distributed Equivalent (EDE)

The EDE represents the level of NHB that, if equally distributed, would give the same social welfare as the current distribution.

$$
EDE = \mu (1 - A_\epsilon)
$$

## 3. Value of Information (VOI)

### Expected Value of Perfect Information (EVPI)

EVPI measures the expected opportunity loss from current uncertainty.

$$
EVPI = E_\theta [\max_j NMB_j(\theta)] - \max_j E_\theta [NMB_j(\theta)]
$$

Where:
- $\theta$ represents the uncertain parameters.
- $j$ indexes the decision options (interventions).
- $E_\theta$ denotes the expectation over the distribution of $\theta$.

### Expected Value of Partial Perfect Information (EVPPI)

EVPPI measures the value of resolving uncertainty for a specific subset of parameters $\phi$ (where $\theta = \{\phi, \psi\}$).

$$
EVPPI_\phi = E_\phi [\max_j E_{\psi|\phi} [NMB_j(\phi, \psi)]] - \max_j E_\theta [NMB_j(\theta)]
$$

## 4. Budget Impact Analysis (BIA)

### Net Budget Impact

$$
BI_t = \sum_{i} (C_{new, i, t} - C_{comparator, i, t}) \times P_{i, t}
$$

Where:
- $BI_t$ is the budget impact in year $t$.
- $C_{i, t}$ is the cost per person in subgroup $i$ in year $t$.
- $P_{i, t}$ is the population size of subgroup $i$ in year $t$.
