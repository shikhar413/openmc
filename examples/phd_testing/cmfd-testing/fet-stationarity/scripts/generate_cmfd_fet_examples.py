import sys
import socket
import os

def generate_input_files(cluster, seed_begin, seed_end, prob_type, run_file):
    if cluster == "Green Cluster":
        param_dict = {
            '1d-homog-offset': {
                'batch_file': "job.slurm",
                'run_command': "sbatch ",
                'base_files': ['1dh-offset-settings.xml', '1dh-geometry.xml', '1dh-materials.xml', 'run_openmc.py'],
                'ppn': 32,
                'nodes': 1,
                'walltime': '12:00:00'
            },
            '2d-beavrs': {
                'batch_file': "job.slurm",
                'run_command': "sbatch ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml', 'run_openmc.py'],
                'ppn': 32,
                'nodes': 1,
                'walltime': '12:00:00',
            }
        }

    elif cluster == "INL Cluster":
        pass

    if prob_type not in param_dict:
        print('Problem type {} not recognized'.format(prob_type))
        sys.exit()
    cluster_params = param_dict[prob_type]

    run_strats = {
        '1d-homog-offset': [
            '0 none',
            '1 0p4cm',
            '1 20cm',
            '2 0p4cm',
            '2 20cm',
            '3 0p4cm',
            '3 20cm',
        ],
        '2d-beavrs': [
            '0 none',
            '1 assembly',
            '1 assembly --refd',
            '1 qassembly',
            '1 qassembly --refd',
            '2 assembly',
            '2 assembly --refd',
            '2 qassembly',
            '2 qassembly --refd',
            '3 assembly',
            '3 assembly --refd',
            '3 qassembly',
            '3 qassembly --refd',
        ]
    }

    os.chdir('./../')

    # Generate file templates
    with open('base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    # Generate files for each problem type
    os.system('mkdir -p {}'.format(prob_type))
    os.chdir(prob_type)

    for seed in range(seed_begin, seed_end+1):
        seed_dir = 'seed{}'.format(str(seed))
        os.system('mkdir -p {}'.format(seed_dir))
        os.chdir(seed_dir)

        for run_strat in run_strats[prob_type]:
            run_dir = 'test'+run_strat.replace('--', '').replace(' ', '-')
            os.system('mkdir -p {}'.format(run_dir))
            os.chdir(run_dir)
            for base_file in cluster_params['base_files']:
                new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                os.system('cp ./../../../base/{} ./{}'.format(base_file, new_file))
            create_files(batch_template, cluster_params, seed, prob_type, run_file, run_strat)
            os.chdir('./..')
        os.chdir('./..')


def create_files(batch_template, cluster_params, seed, prob_name, run_file, run_strat):
    os.system("sed -i 's-{seed}"+"-{}-g' settings.xml".format(seed))
    test_num = run_strat.split(' ')[0]
    jobname = 's{}-r{}'.format(seed, test_num)
    nodes = cluster_params['nodes']
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{nproc}', str(nprocs))
    openmc_args = "{} {} {}".format(nthreads, prob_name, run_strat)
    batch_template = batch_template.replace('{openmc_args}', openmc_args)
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    print_str = os.getcwd().split('fet-stationarity/')[-1]
    if run_file:
        print('Running batch script in {}'.format(print_str))
        os.system(cluster_params['run_command']+cluster_params['batch_file'])
    else:
        print('Created input files in {}'.format(print_str))

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
    if len(sys.argv) not in [4, 5]:
        print('Usage: generate_cmfd_fet_examples.py [seed_begin] [seed_end] [prob_type] [-r]')
        sys.exit()

    # Get command line arguments
    seed_begin = int(sys.argv[1])
    seed_end = int(sys.argv[2])
    prob_type = sys.argv[3]
    run_file = len(sys.argv) == 5 and sys.argv[4] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, seed_begin, seed_end, prob_type, run_file)
