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
        batch_file = "job-nse.qsub"
        run_command = "qsub "
        xml_files = ['1dh-offset-settings.xml', '1dh-geometry.xml', '1dh-materials.xml']
        ppn = 12
        nodes = 5
        walltime = '02:00:00'
    elif cluster == "Green Cluster":
        prob_type = "2d-beavrs"
        batch_file = "job.slurm"
        run_command = "sbatch "
        xml_files = ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml']
        ppn = 32
        nodes = 1
        walltime = '12:00:00'

    # Get problem parameters
    params, ds = get_params(prob_type)

    os.system('mkdir -p ./../tally_solver_begin_sensitivity')
    os.chdir('./../tally_solver_begin_sensitivity')

    # Generate files for base run
    with open('../base/run_openmc_cmfd_ts_begin.py', 'r') as file:
        cmfd_template = file.read()
    with open('../base/{}'.format(batch_file), 'r') as file:
        batch_template = file.read() 

    os.system('mkdir -p {}'.format(prob_type))
    os.chdir(prob_type)

    for seed in range(1, n_seeds+1):
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
                tasks = nodes * ppn
                nprocs = nodes*2
                nthreads = int(ppn/2)

                batch_out = batch_template
                batch_out = batch_out.replace('{prob_type}', prob_type)
                batch_out = batch_out.replace('{tasks}', str(tasks))
                batch_out = batch_out.replace('{nodes}', str(nodes))
                batch_out = batch_out.replace('{ppn}', str(ppn))
                batch_out = batch_out.replace('{job_name}', jobname)
                batch_out = batch_out.replace('{walltime}', walltime)
                batch_out = batch_out.replace('{nprocs}', str(nprocs))
                batch_out = batch_out.replace('{nthreads}', str(nthreads))

                with open('run_openmc_cmfd.py', "w") as file:
                    file.write(cmfd_out)
                with open(batch_file, "w") as file:
                    file.write(batch_out)
                for xml_file in xml_files:
                    os.system('cp ./../../../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))

                print_str = os.getcwd().split('tally_solver_begin_sensitivity/')[-1]
                if run_file:
                    print('Running batch script in {}'.format(print_str))
                    os.system(run_command+batch_file)
                else:
                    print('Created input files in {}'.format(print_str))
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
    n_seeds = int(sys.argv[1])
    run_file = len(sys.argv) == 3 and sys.argv[2] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, n_seeds, run_file)
