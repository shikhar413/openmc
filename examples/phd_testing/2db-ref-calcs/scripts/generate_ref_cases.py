import sys
import socket
import os

def generate_input_files(seed_begin, seed_end, prob_type, run_strat, run_file):
    seed_offset = 10000000
    cluster_params = {
        'batch_file': "job-inl.qsub",
        'run_command': "qsub ",
        'base_files': ['settings.xml', 'geometry.xml', 'materials.xml', 'run_openmc.py'],
        'ppn': 36,
        'nodes': 10,
        'walltime': {'1d-homog': '24:00:00',
                     '2d-beavrs': '12:00:00'}
    }
    strat_dirname_dict = {
        0: 'nocmfd',
        1: 'cmfd-qassembly',
        2: 'cmfd-0p4cm'
    }
    if prob_type == '1d-homog':
        prob_prefix = '1dh'
    elif prob_type == '2d-beavrs':
        prob_prefix = '2db'

    os.chdir('./../')

    # Generate file templates
    with open('base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    # Generate files for each problem type
    prob_dirname = '{}-{}'.format(prob_type, strat_dirname_dict[run_strat])
    if prob_dirname not in ['1d-homog-nocmfd', '1d-homog-cmfd-0p4cm', '2d-beavrs-cmfd-qassembly']:
        print('Invalid combination of problem type and run strategy')
        sys.exit()

    os.system('mkdir -p {}'.format(prob_dirname))
    os.chdir(prob_dirname)

    for seed in range(seed_begin, seed_end+1):
        seed_dir = 'seed{}'.format(str(seed))
        os.system('mkdir -p {}'.format(seed_dir))
        os.chdir(seed_dir)
        for base_file in cluster_params['base_files']:
            filename = '{}-{}'.format(prob_prefix, base_file) if 'xml' in base_file else base_file
            os.system('cp ./../../base/{} ./{}'.format(filename, base_file))
        os.system("sed -i 's-{seed}"+"-{}-g' settings.xml".format(seed+seed_offset))
        create_files(prob_prefix, batch_template, cluster_params, seed, prob_type, run_file, run_strat)

        os.chdir('./..')
    os.chdir('./..')

def create_files(prob_prefix, batch_template, cluster_params, seed, prob_name, run_file, run_strat):
    jobname = '{}s{}r{}'.format(prob_prefix, seed, run_strat)
    nodes = cluster_params['nodes']
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'][prob_type])
    batch_template = batch_template.replace('{nprocs}', str(nprocs))
    openmc_args = "{} {} {}".format(nthreads, prob_name, run_strat)
    batch_template = batch_template.replace('{openmc_args}', openmc_args)
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    print_str = os.getcwd().split('2db-ref-calcs/')[-1]
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
    if len(sys.argv) not in [5, 6]:
        print('Usage: generate_ref_cases.py [seeds_begin] [seed_end] [prob_type] [run_strat] [-r]')
        sys.exit()

    # Get command line arguments
    seed_begin = int(sys.argv[1])
    seed_end = int(sys.argv[2])
    prob_type = sys.argv[3]
    run_strat = int(sys.argv[4])
    run_file = len(sys.argv) == 6 and sys.argv[5] == '-r'

    # Generate OpenMC and batch script input files
    generate_input_files(seed_begin, seed_end, prob_type, run_strat, run_file)
