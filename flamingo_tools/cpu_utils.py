import multiprocessing as mp
import os


def available_cpus() -> int:
    """Return the number of cores this process may use.

    `multiprocessing.cpu_count` reports the cores of the whole node, which oversubscribes
    a job on a shared slurm partition. Prefer the slurm allocation and the CPU affinity mask.

    Returns:
        The number of usable cores.
    """
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm is not None:
        return int(n_slurm)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return mp.cpu_count()
