import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit


SAVE_FIGURES = False # False: show, True: save


def fit_exp_and_power(r: np.ndarray, ri: float, p: float, A: float) -> np.ndarray:
    """f(r) = A * exp(-r/ri) / r^p  for r > 0"""
    return A * np.exp(-r / ri) / r**p


def log_fit_exp_and_power(r: np.ndarray, ri: float, p: float, log_A: float) -> np.ndarray:
    """log(f(r)) = log_A - r/ri - p*log(r)"""
    return log_A - r / ri - p * np.log(r)


def show_or_save(filename: str) -> None:
    if SAVE_FIGURES:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    lattice_length = 2000
    csv_path = Path(__file__).parents[2] / "data" / f"ising_correlation_{lattice_length}x{lattice_length}.csv"
    df = pd.read_csv(csv_path)

    temperatures = sorted(set(df["temperature"]))
    df = df[(df["r"] > 0) & (df["r"] < 10) & (df["C_r"] > 0)]

    ri_values, p_values = [], []

    # --- Plot 1: correlation data + fits ---
    plt.figure(figsize=(8, 5.5))
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

        plt.plot(r, C_r, marker="o", linestyle="", label=f"T={temp:.3f}")
        plt.plot(
            r,
            fit_exp_and_power(r, ri_fit, p_fit, np.exp(log_A_fit)),
            linestyle="--",
            label=f"Fit T={temp:.3f}, ri={ri_fit:.1e}, p={p_fit:.3f}",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance r")
    plt.ylabel("Correlation C(r)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    show_or_save(f"figures/plots/correlation_fit_{lattice_length}x{lattice_length}.png")

    # --- Save results ---
    df_temp_corr = pd.DataFrame({"temperature": temperatures, "correlation_length_scale": ri_values, "power": p_values})
    df_temp_corr.to_csv(f"data/temp_correlation_{lattice_length}x{lattice_length}.csv", index=False)
    print("Saved temp corrleaiton data to CSV file")

    # --- Plot 2: correlation length vs T ---
    plt.figure(figsize=(6, 4))
    plt.plot(temperatures, ri_values, marker="o", label="Correlation Length (ri)")
    plt.xlabel("Temperature T")
    plt.ylabel("Correlation Length")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    show_or_save(f"figures/plots/correlation_length_vs_temp_{lattice_length}x{lattice_length}.png")

    # --- Plot 3: power law exponent vs T ---
    plt.figure(figsize=(6, 4))
    plt.plot(temperatures, p_values, marker="o", label="Power Law Exponent (p)")
    plt.hlines(0.25, temperatures[0], temperatures[-1], colors="r", linestyles="--", label="p=0.25 (expected at Tc)")
    plt.xlabel("Temperature T")
    plt.ylabel("Power Law Exponent")
    plt.ylim(0, max(p_values) * 1.5)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    show_or_save(f"figures/plots/power_law_exponent_vs_temp_{lattice_length}x{lattice_length}.png")
