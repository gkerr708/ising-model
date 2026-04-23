import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
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


def add_temp_colorbar(fig, ax, norm, cmap) -> None:
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Temperature T")


if __name__ == "__main__":
    lattice_length = 100
    n_sweeps = 50_000
    csv_path = Path(__file__).parents[2] / "data" / f"ising_correlation_{lattice_length}x{lattice_length}_N{n_sweeps}.csv"
    df = pd.read_csv(csv_path)

    temperatures = sorted(set(df["temperature"]))
    df = df[(df["r"] > 0) & (df["r"] < 10) & (df["C_r"] > 0)]

    cmap = cm.coolwarm
    norm = mcolors.Normalize(vmin=min(temperatures), vmax=max(temperatures))

    ri_values, p_values = [], []

    # --- Plot 1: correlation data + fits ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for temp in temperatures:
        color = cmap(norm(temp))
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

        ax.plot(r, C_r, marker="o", linestyle="", color=color)
        ax.plot(
            r,
            fit_exp_and_power(r, ri_fit, p_fit, np.exp(log_A_fit)),
            linestyle="--",
            color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance r")
    ax.set_ylabel("Correlation C(r)")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="gray", linestyle="", label="Data"),
        Line2D([0], [0], color="gray", linestyle="--", label="Fit"),
    ])
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    add_temp_colorbar(fig, ax, norm, cmap)
    show_or_save(f"figures/plots/correlation_fit_{lattice_length}x{lattice_length}.png")

    # --- Save results ---
    df_temp_corr = pd.DataFrame({"temperature": temperatures, "correlation_length_scale": ri_values, "power": p_values})
    df_temp_corr.to_csv(f"data/temp_correlation_{lattice_length}x{lattice_length}.csv", index=False)
    print("Saved temp corrleaiton data to CSV file")

    # --- Plot 2: correlation length vs T ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temperatures, ri_values, color="gray", linewidth=0.8, zorder=2)
    ax.scatter(temperatures, ri_values, c=temperatures, cmap=cmap, norm=norm, zorder=3)
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Correlation Length")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    add_temp_colorbar(fig, ax, norm, cmap)
    show_or_save(f"figures/plots/correlation_length_vs_temp_{lattice_length}x{lattice_length}.png")

    # --- Plot 3: power law exponent vs T ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temperatures, p_values, color="gray", linewidth=0.8, zorder=2)
    ax.scatter(temperatures, p_values, c=temperatures, cmap=cmap, norm=norm, zorder=3)
    ax.hlines(0.25, temperatures[0], temperatures[-1], colors="black", linestyles="--", label="p=0.25 (expected at Tc)")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Power Law Exponent")
    ax.set_ylim(0, max(p_values) * 1.5)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    add_temp_colorbar(fig, ax, norm, cmap)
    show_or_save(f"figures/plots/power_law_exponent_vs_temp_{lattice_length}x{lattice_length}.png")
