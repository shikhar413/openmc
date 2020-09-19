import sys
import socket
import os

def generate_input_files(cluster, prob_type, run_file):
    if cluster == "INL Cluster":
        param_dict = {
            '1d-homog': {
                'batch_file': "job-inl.qsub",
                'params_file': '1dh-params.cfg',
                'partition': 'neup',
                'run_command': "qsub ",
                'base_files': ['1dh-unif-settings.xml', '1dh-geometry.xml', '1dh-materials.xml'],
                'tasks_per_node': 36,
                'procs_per_node': 2,
                'walltime': '12:00:00',
                'mult_factor': 16,
                'n_particles': 10000000,
                'n_batches': 499,
                'max_window_size': 64
            },
            '2d-beavrs': {
                'batch_file': "job-inl.qsub",
                'params_file': '2db-params.cfg',
                'partition': 'neup',
                'run_command': "qsub ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml'],
                'tasks_per_node': 36,
                'procs_per_node': 2,
                'walltime': '12:00:00',
                'mult_factor': 16,
                'n_particles': 10000000,
                'n_batches': 199,
                'max_window_size': 16
            }
        }

    if prob_type not in param_dict:
        print('Problem type {} not recognized'.format(prob_type))
        sys.exit()
    cluster_params = param_dict[prob_type]

    os.chdir('./../')

    # Generate file templates
    with open('base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    with open('base/{}'.format(cluster_params['params_file']), 'r') as file:
        params_template = file.read()

    os.system('mkdir -p var_testing')
    os.chdir('var_testing')
    # Generate files for each problem type
    os.system('mkdir -p {}'.format(prob_type))
    os.chdir(prob_type)

    run_strats = [
        {'particle_mult': False, 'seed_mult': True, 'test_num': 0, 'ea_run': False},                   # 16seed-10M-noCMFD
        {'particle_mult': False, 'seed_mult': True, 'test_num': 2, 'ea_run': False},                   # 16seed-10M-CMFD-window64
        {'particle_mult': False, 'seed_mult': True, 'ea_run': True, 'ea_run_strat': 'bulk-synch'},    # 16seed-10M-bsCMFD-window64
        {'particle_mult': False, 'seed_mult': True, 'ea_run': True, 'ea_run_strat': 'rendez-asynch'},    # 16seed-10M-raCMFD-window64
        {'particle_mult': True, 'seed_mult': False, 'test_num': 0, 'ea_run': False},     # 1seed-160M-noCMFD
        {'particle_mult': True, 'seed_mult': False, 'test_num': 2, 'ea_run': False}  # 1seed-160M-CMFD-window64
    ]

    for run_strat in run_strats:
        run_strat['n_particles'] = cluster_params['mult_factor']*cluster_params['n_particles'] if run_strat['particle_mult'] else cluster_params['n_particles']
        run_strat['n_seeds'] = cluster_params['mult_factor'] if run_strat['seed_mult'] else 1

        if 'test_num' in run_strat:
            run_type = 'noCMFD' if run_strat['test_num'] == 0 else 'CMFD-window{}'.format(cluster_params['max_window_size'])
        else:
            run_type = '{}-window{}'.format(run_strat['ea_run_strat'], cluster_params['max_window_size'])

        ea_run = run_strat['ea_run']
        dir_name = '{}seed-{}M-{}'.format(run_strat['n_seeds'], int(run_strat['n_particles']/1000000), run_type)
        run_strat['name'] = dir_name
        run_strat['nodes'] = cluster_params['mult_factor']+1 if ea_run else cluster_params['mult_factor']
        os.system('mkdir -p {}'.format(dir_name))
        os.chdir(dir_name)
        for base_file in cluster_params['base_files']:
            new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
            os.system('cp ./../../../base/{} ./{}'.format(base_file, new_file))

        if ea_run:
            os.system('cp ./../../../base/run_openmc_ea.py ./run_openmc.py')
        else:
            os.system('cp ./../../../base/run_openmc_noea.py ./run_openmc.py')
             
        create_files(batch_template, params_template, cluster_params, prob_type, run_file, run_strat)
        os.chdir('./..')
    os.chdir('./..')

def create_files(batch_template, params_template, cluster_params, prob_name, run_file, run_strat):
    n_batches = cluster_params['n_batches']
    n_inactive = n_batches - 1
    n_particles = 1 if run_strat['ea_run'] else run_strat['n_particles']
    os.system("sed -i 's-{n_particles}"+"-{}-g' settings.xml".format(n_particles))
    os.system("sed -i 's-{n_batches}"+"-{}-g' settings.xml".format(n_batches))
    os.system("sed -i 's-{n_inactive}"+"-{}-g' settings.xml".format(n_inactive))
    jobname = prob_name
    nodes = run_strat['nodes']
    tasks = nodes * cluster_params['tasks_per_node']
    procs_per_node = cluster_params['procs_per_node']
    nprocs = nodes * procs_per_node
    threads_per_proc = int(cluster_params['tasks_per_node']/procs_per_node)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{tasks_per_node}', str(cluster_params['tasks_per_node']))
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{partition}', cluster_params['partition'])
    batch_template = batch_template.replace('{procs_per_node}', str(procs_per_node))
    batch_template = batch_template.replace('{nproc}', str(nprocs))
    if run_strat['ea_run']:
        openmc_args = prob_name
        params_template = params_template.replace('{n_seeds}', str(run_strat['n_seeds']))
        params_template = params_template.replace('{n_procs_per_seed}', str(procs_per_node))
        params_template = params_template.replace('{ea_run_strat}', run_strat['ea_run_strat'])
        params_template = params_template.replace('{max_window_size}', str(cluster_params['max_window_size']))
        params_template = params_template.replace('{n_particles}', str(run_strat['n_particles']))
        params_template = params_template.replace('{threads_per_proc}', str(threads_per_proc))
        params_template = params_template.replace('{n_batches}', str(n_batches))
        params_template = params_template.replace('{n_inactive}', str(n_inactive))
        params_template = params_template.replace('{use_logger}', 'false')
        with open('params.cfg', 'w') as f:
            f.write(params_template)

    else:
        max_window_size = cluster_params['max_window_size'] if run_strat['test_num'] == 2 else 0
        openmc_args = "{} {} {} {} --no-log {}".format(run_strat['n_seeds'], threads_per_proc, prob_name, run_strat['test_num'], max_window_size)
    batch_template = batch_template.replace('{openmc_args}', openmc_args)
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    print_str = os.getcwd().split('ensemble-averaging/')[-1]
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
    elif socket_name.startswith('bebop'):
        return "LCRC Cluster"
    else:
        print('Unrecognized socket name')
        sys.exit()

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print('Usage: generate_cmfd_fet_examples.py [prob_type] [-r]')
        sys.exit()

    # Get command line arguments
    prob_type = sys.argv[1]
    run_file = len(sys.argv) == 3 and sys.argv[2] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, prob_type, run_file)
