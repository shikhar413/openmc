import openmc.lib
from openmc.ensemble_averaging import EnsAvgCMFDRun
from openmc import cmfd

if __name__ == "__main__":
    # Initialize and run CMFDRun object
    ea_cmfd_run = EnsAvgCMFDRun()
    ea_cmfd_run.cfg_file = "params.cfg"
    with ea_cmfd_run.run_in_memory():
        for _ in ea_cmfd_run.iter_batches():
            pass
