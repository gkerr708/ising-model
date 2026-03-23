import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit


def fit_exp_and_power(r: np.ndarray, ri: float, p: float, A: float) -> np.ndarray:
    """
    f(r) = A * exp(-r/ri) / r^p  for r > 0
    """
    return A * np.exp(-r / ri) / r**p


def log_fit_exp_and_power(r: np.ndarray, ri: float, p: float, log_A: float) -> np.ndarray:
    """
    log(f(r)) = log_A - r/ri - p*log(r)
    Fitting in log-space gives equal weight to all decades on the log-log plot.
    """
    return log_A - r / ri - p * np.log(r)



if __name__ == "__main__":
    
    # Example usage
    csv_path = Path(__file__).parents[2] / "ising_correlation_500x500.csv"
    df = pd.read_csv(csv_path)
    temperatures = sorted(set(df["temperature"]))
    max_r = 10
    df = df[(df["r"] > 0) & (df["r"] < max_r) & (df["C_r"] > 0)]

    plt.figure(figsize=(8,5.5))
    for temp in temperatures:
        subset = df[df["temperature"] == temp]
        r = subset["r"].values
        C_r = subset["C_r"].values
        popt, _ = curve_fit(
            log_fit_exp_and_power,
            r,
            np.log(C_r),
            p0=(10.0, 1.0, np.log(C_r[0])),
            maxfev=10_000,
        )
        ri_fit, p_fit, log_A_fit = popt
        popt_orig = (ri_fit, p_fit, np.exp(log_A_fit))

        plt.plot(r, C_r, marker="o", linestyle="", label=f"T={temp:.3f}")
        plt.plot(
            r,
            fit_exp_and_power(r, *popt_orig),
            linestyle="--",
            label=f"Fit T={temp:.3f}, ri={ri_fit:.1e}, p={p_fit:.3f}",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance r")
    plt.ylabel("Correlation C(r)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.savefig("correlation_fit.png", dpi=300)

    # How plot ri vs temp and p vs temp on the other axis
    plt.figure(figsize=(6, 4))
    ri_values = []
    p_values = []
    for temp in temperatures:
        subset = df[df["temperature"] == temp]
        r = subset["r"].values
        C_r = subset["C_r"].values
        popt, _ = curve_fit(
            log_fit_exp_and_power,
            r,
            np.log(C_r),
            p0=(10.0, 1.0, np.log(C_r[0])),
            maxfev=10_000,
        )
        ri_fit, p_fit, log_A_fit = popt
        ri_values.append(ri_fit)
        p_values.append(p_fit)

    plt.plot(temperatures, ri_values, marker="o", label="Correlation Length (ri)")
    plt.xlabel("Temperature T")
    plt.ylabel("Fitted Parameters")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.savefig("correlation_length_vs_temp.png", dpi=300)
    plt.close()


    plt.figure(figsize=(6,4))
    plt.plot(temperatures, p_values, marker="o", label="Power Law Exponent (p)")
    plt.hlines(0.25, temperatures[0], temperatures[-1], colors="r", linestyles="--", label="p=0.25 (expected at Tc)")
    plt.xlabel("Temperature T")
    plt.ylim(0, max(p_values) * 1.5)
    plt.ylabel("Fitted Parameters")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.savefig("power_law_exponent_vs_temp.png", dpi=300)



