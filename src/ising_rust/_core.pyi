def ising_sim(
    rows: int,
    cols: int,
    temperature: float,
    n_therm: int,
    n_sweeps: int,
    seed: int | None = None,
) -> list[int]:
    """
    Simulate the 2D Ising model using the Metropolis algorithm.
    Returns the final lattice state:
        list[int]: A list of integers representing the final state of the lattice, where each integer is either -1 or 1.
    """
    ...
