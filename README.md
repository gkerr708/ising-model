# Ising Model Snapshots and Correlations
 
### To-Do
- [ ] Output a csv with the correlation function results.

### Low Temperature (Ordered Phase)
<p align="center">
<img src="ising_snapshot_T0.769_N1000_L2000.png" width="450">
</p>

Large aligned spin domains appear.

---

### Near Critical Temperature \(T_c\)
<p align="center">
<img src="ising_snapshot_T2.269_N1000_L2000.png" width="450">
</p>

Fluctuations occur at many length scales.

---

### Above \(T_c\) (Disordered Phase)
<p align="center">
<img src="ising_snapshot_T2.769_N1000_L2000.png" width="450">
</p>

Spins appear mostly random with short-range correlations.

---
# Modeling
Using $500 \times 500$ lattice.

### Correlation Fit
<p align="center">
<img src="correlation_fit.png" width="650">
</p>
Fit using:

$$
C(r) = \frac{Ae^{-r/\xi}}{r^{\eta}}.
$$

### Correlation Length vs Temperature
<p align="center">
<img src="correlation_length_vs_temp.png" width="450">
</p>

The correlation length $\xi$ diverges as $T$ approaches $T_c$.

### Power-Law Exponent vs Temperature
<p align="center">
<img src="power_law_exponent_vs_temp.png" width="450">
</p>
 
Should approach $p \approx 0.25$ at $T_c$.




