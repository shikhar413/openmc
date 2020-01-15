import sys
import socket
import os

def get_params(prob_type):
    if prob_type == '1d-homog':
        vary_params = [[2,10], [2,12], [2,14], [2,16], [2,18], [2,20], [4,20], [6,20], [8,20],
                       [8,20], [10,20], [10,10], [10,12], [10,14], [10,16], [10,18], [4,10],
                       [6,10], [8,10]]
        ref_ds = ['']
    elif prob_type == '2d-beavrs':
        vary_params = [[2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9], [2,10],
                       [3,10], [4,10], [5,10], [6,10], [7,10], [8,10], [9,10]]
        ref_ds = ['', '1.42669,0.400433']

    return vary_params, ref_ds


def generate_input_files(cluster, n_seeds, run_file):
    if cluster == "NSE Cluster":
        prob_type = "1d-homog"
        batch_file = "job.qsub"
        run_command = "qsub "
    elif cluster == "Green Cluster":
        prob_type = "2d-beavrs"
        batch_file = "job.slurm"
        run_command = "sbatch "

    # Get problem parameters
    params, ds = get_params(prob_type)

    os.chdir('./../{}'.format(prob_type))

    # Generate files for base run
    with open('base/run_openmc_cmfd_ts_begin.py', 'r') as file:
        cmfd_template = file.read()
    with open('base/{}'.format(batch_file), 'r') as file:
        runfile = file.read() 

    os.system('mkdir -p vary_tally_solver_begin')
    os.chdir('vary_tally_solver_begin')

    for seed in n_seeds:
        seed_dir = 'seed{}'.format(str(seed))
        os.system('mkdir -p {}'.format(seed_dir))
        os.chdir(seed_dir)

        for ref_d in ds:
            if ref_d == '':
                d_dir = 'no_refd'
            else:
                d_dir = 'refd'
            os.system('mkdir -p {}'.format(d_dir))
            os.chdir(d_dir)

            for comb in params:
                comb_dir = 'tbeg{}_sbeg{}'.format(comb[0], comb[1])
                os.system('mkdir -p {}'.format(comb_dir))
                os.chdir(comb_dir)
                cmfd_out = cmfd_template

                cmfd_out = cmfd_out.replace('{seed}', str(seed))
                cmfd_out = cmfd_out.replace('{tbeg}', str(comb[0]))
                cmfd_out = cmfd_out.replace('{ref_d}', '{}'.format(ref_d))
                cmfd_out = cmfd_out.replace('{sbeg}', str(comb[1]))

                jobname = 'tb{}sb{}s{}'.format(comb[0], comb[1], seed)
                runfile = runfile.replace('{job_name}', jobname)

                with open('run_openmc_cmfd.py', "w") as file:
                    file.write(cmfd_out)
                with open(batch_file, "w") as file:
                    file.write(runfile)
                os.system("cp ../../../../base/*.xml ./")

                if run_file:
                    print('Running batch script in {}/{}/{}'.format(seed_dir, d_dir, comb_dir))
                    os.system(run_command+batch_file)
                else:
                    print('Created input files for {}/{}/{}'.format(seed_dir, d_dir, comb_dir))
                os.chdir('./..')
            os.chdir('./..')
        os.chdir('./..')
    os.chdir('./..')


def get_cluster(socket_name):
    if socket_name.startswith('nsecluster'):
        return "NSE Cluster"
    elif socket_name.startswith('eofe'):
        return "Green Cluster"
    elif socket_name.startswith('sk-linux'):
        return "SK Linux"
    elif socket_name.startswith('falcon'):
        return "INL Cluster"
    else:
        print('Unrecognized socket name')
        sys.exit()

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print('Usage: tally_solver_begin_sensitivity.py [n_seeds] [-r]')
        sys.exit()

    # Get command line arguments
    n_seeds = sys.argv[1]
    run_file = len(sys.argv) == 3 and sys.argv[2] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, n_seeds, run_file)
