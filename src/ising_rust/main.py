from ._core import ising_sim
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.optimize import curve_fit
import pandas as pd


def autocorr2d_periodic(
    lattice: np.ndarray,
    *,
    subtract_mean: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Periodic 2D autocorrelation via FFT
    Returns array with zero-shift at [0,0] (i.e., not fftshifted).

    lattice: (rows, cols) array (e.g., spins +/-1)
    """
    s = np.asarray(lattice, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"lattice must be 2D, got shape {s.shape}")

    if subtract_mean:
        s = s - s.mean()

    F = fft2(s)
    corr = ifft2(F * np.conj(F)).real  # type: ignore

    if normalize:
        corr /= s.size

    return corr


def autocorr_radial(
    lattice: np.ndarray,
    *,
    max_r: int | None = None,
    subtract_mean: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radially average C(r) from periodic 2D autocorrelation.

    Returns:
      r_centers: (max_r,) array of integer radii [0,1,2,...]
      C_r:       (max_r,) radial average of C2D in annuli [r, r+1)

    Note: uses fftshift to put zero-lag at the center before binning.
    """
    C2 = autocorr2d_periodic(lattice, subtract_mean=subtract_mean, normalize=normalize)
    C2c = np.fft.fftshift(C2)

    rows, cols = C2c.shape
    y, x = np.indices(C2c.shape)
    cy, cx = rows // 2, cols // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    if max_r is None:
        max_r = min(rows, cols) // 2

    if max_r is None:
        raise ValueError(f"max_r must be a positive integer, got {max_r}")

    C_r = np.empty(max_r, dtype=np.float64)
    for rr in range(max_r):
        mask = (r >= rr) & (r < rr + 1)
        C_r[rr] = C2c[mask].mean() if np.any(mask) else np.nan

    r_centers = np.arange(max_r, dtype=np.float64)
    return r_centers, C_r


def autocorr_radial_avg_over_snapshots(
    lattices: list[np.ndarray] | np.ndarray,
    *,
    max_r: int,
    subtract_mean: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: average radial C(r) over many snapshots.

    lattices can be:
      - list of (rows, cols) arrays, or
      - array of shape (n, rows, cols)

    Returns (r, C_avg).
    """
    if isinstance(lattices, list):
        snaps = lattices
    else:
        arr = np.asarray(lattices)
        if arr.ndim != 3:
            raise ValueError(f"Expected (n, rows, cols), got {arr.shape}")
        snaps = [arr[i] for i in range(arr.shape[0])]

    C_sum = np.zeros(max_r, dtype=np.float64)
    r_out: np.ndarray | None = None

    for lat in snaps:
        r, C = autocorr_radial(
            lat,
            max_r=max_r,
            subtract_mean=subtract_mean,
            normalize=normalize,
        )
        C_sum += C
        r_out = r

    if r_out is None:
        raise ValueError("No snapshots provided")

    return r_out, C_sum / len(snaps)

def save_image(
    lattice_2d: np.ndarray, 
    temperature: float, 
    n_sweeps: int, 
    rows: int, cols: int
) -> None:
    """
    """
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(lattice_2d, vmin=-1, vmax=1, interpolation="nearest", cmap="Greys")
    ax.set_title(f"Ising snapshot (T={temperature:.3f}, L={rows}, sweeps={n_sweeps})")
    ax.set_axis_off()
    fig.savefig(f"ising_snapshot_T{temperature:.3f}_N{n_sweeps}_L{rows}.png", dpi=300)
    plt.close(fig)


def power_law(r: np.ndarray, A: float, xi: float) -> np.ndarray:
    """
    Power-law decay function for fitting 
    y = A r^{-1/xi}
    """
    return A * r ** (-1 / xi)

def main() -> None:


    # Parameters
    print("Running Ising simulation...")
    rows = 2000
    cols = 2000
    tc = 2.26918531421347
    #temps = [tc - 1.5, tc, tc + 0.5]
    temps = [tc, 
             tc+0.01, 
             tc+0.05, 
             tc+0.1, 
             tc+0.2,
             tc+0.5,
             ]
    n_therm = 2  # this currently does nothing
    n_sweeps = 1000
    seed = None

    C_r_list = []
    r_list = []
    fit_list: list[tuple[float, float, float, float] | None] = []  # (A, xi, R^2, temperature)

    max_r = None # num points for fit

    dataset = []

    # Run simulations and analyze correlations
    for temperature in temps:
        lattice: list[int] = ising_sim(rows, cols, temperature, n_therm, n_sweeps, seed)

        # Faster reshape than python slicing
        lattice_2d = np.asarray(lattice, dtype=np.int8).reshape(rows, cols)

        # Snapshot
        #save_image(lattice_2d, temperature, n_sweeps, rows, cols)

        # Correlation
        r, C_r = autocorr_radial(lattice_2d, max_r=max_r)

        # drop r=0 and any nonpositive entries for log fits
        fit_mask = (r > 0) & np.isfinite(C_r) & (C_r > 0)

        if np.sum(fit_mask) > 6:
            # Fit only on a mid-range to avoid tiny-r lattice effects and noisy tail
            r_fit = r[fit_mask]
            C_fit = C_r[fit_mask]

            # heuristic window: ignore first few points; keep up to where C falls to noise-ish
            lo = 1
            hi = min(len(r_fit), 60)
            r_fit2 = r_fit[lo:hi]
            C_fit2 = C_fit[lo:hi]

            popt, _ = curve_fit(power_law, r_fit2, C_fit2, p0=(1.0, 1.0), maxfev=10_000)
            A_fit, xi_fit = popt

            residual = C_fit2 - power_law(r_fit2, *popt)
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((C_fit2 - np.mean(C_fit2)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            print(f"T={temperature:.3f}: A={A_fit:.3g}, xi={xi_fit:.3g}, R^2={r_squared:.3f}")
            fit_list.append((A_fit, xi_fit, r_squared, temperature))
        else:
            fit_list.append(None)

        C_r_list.append(C_r)
        r_list.append(r)
        dataset.append({"temperature": temperature, "r": r, "C_r": C_r})


    # Save data
    df = pd.concat(
        [pd.DataFrame({"temperature": d["temperature"], "r": d["r"], "C_r": d["C_r"]}) for d in dataset],
        ignore_index=True,
    )
    df.to_csv(f"ising_correlation_{rows}x{cols}.csv", index=False)

    # Generate plots for all temperatures
    # This should be moved to all different file since we are saving this to a CSV
    #fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    #colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    #for i, (temperature, r, C_r) in enumerate(zip(temps, r_list, C_r_list)):

    #    # remove r=0 and invalid values
    #    mask = (r > 0) & np.isfinite(C_r) & (C_r > 0)
    #    r_valid = r[mask]
    #    C_valid = C_r[mask]

    #    # take first fit_points points
    #    fit_points = 10
    #    r_fit = r_valid[:fit_points]
    #    C_fit = C_valid[:fit_points]

    #    if len(r_fit) >= 3:
    #        popt, _ = curve_fit(power_law, r_fit, C_fit, p0=(1.0, 1.0))
    #        A_fit, xi_fit = popt

    #        residual = C_fit - power_law(r_fit, *popt)
    #        ss_res = np.sum(residual**2)
    #        ss_tot = np.sum((C_fit - np.mean(C_fit))**2)
    #        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    #        print(f"T={temperature:.3f}: xi={xi_fit:.3f}, R^2={r_squared:.3f}")

    #        ax.plot(
    #            r_fit,
    #            power_law(r_fit, *popt),
    #            color=colors[i],
    #            linestyle="--",
    #            linewidth=2,
    #            #label=f"Fit T={temperature:.3f}",
    #        )

    #    ax.plot(
    #        r_fit,
    #        C_fit,
    #        color=colors[i],
    #        marker="o",
    #        markersize=6,
    #        linewidth=1.5,
    #        label=f"T={temperature:.3f}",
    #    )

    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #
    #ax.set_xlabel("Distance $r$")
    #ax.set_ylabel("Correlation $C(r)$")
    ##ax.set_title("Ising Radial Correlation Near $T_c$")
    #
    #ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    #ax.legend()
    #
    ##fig.savefig("ising_autocorr_high-to-tc.png", dpi=300, bbox_inches="tight")
    #plt.close(fig)




if __name__ == "__main__":
    main()
