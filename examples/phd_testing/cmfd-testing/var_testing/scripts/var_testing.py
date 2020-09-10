import sys
import socket
import os

def generate_input_files(cluster, prob_type, run_file):
    if cluster == "INL Cluster":
        param_dict = {
            '1d-homog': {
                'batch_file': "job-inl.qsub",
                'run_command': "qsub ",
                'base_files': ['1dh-unif-settings.xml', '1dh-geometry.xml', '1dh-materials.xml', 'run_openmc.py'],
                'ppn': 36,
                'nodes': 4,
                'walltime': '48:00:00',
                'partition': 'neup'
            },
            '2d-beavrs': {
                'batch_file': "job-inl.qsub",
                'run_command': "qsub ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml', 'run_openmc.py'],
                'ppn': 36,
                'nodes': 1,
                'walltime': '48:00:00',
                'partition': 'neup'
            }
        }

    elif cluster == "LCRC Cluster":
        param_dict = {
            '2d-beavrs': {
                'batch_file': "job.slurm",
                'run_command': "sbatch ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml', 'run_openmc.py'],
                'ppn': 36,
                'nodes': 1,
                'walltime': '24:00:00',
                'partition': 'bdwall'
            }
        }

    if prob_type not in param_dict:
        print('Problem type {} not recognized'.format(prob_type))
        sys.exit()
    cluster_params = param_dict[prob_type]

    run_strats = {
        '2d-beavrs': [
            'nocmfd',
            'pincell nolim',
            'pincell 16',
            'pincell 32',
            'pincell 64',
            'pincell 128',
            'qassembly nolim',
            'qassembly 16',
            'qassembly 32',
            'qassembly 64',
            'qassembly 128',
            'assembly nolim',
            'assembly 16',
            'assembly 32',
            'assembly 64',
            'assembly 128',
        ],
        '1d-homog': [
            'nocmfd',
            '0p4cm nolim',
            '0p4cm 16',
            '0p4cm 32',
            '0p4cm 64',
            '0p4cm 128',
            '20cm nolim',
            '20cm 16',
            '20cm 32',
            '20cm 64',
            '20cm 128',
        ]
    }

    os.chdir('./../')

    # Generate file templates
    with open('base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    # Generate files for each problem type
    os.system('mkdir -p {}'.format(prob_type))
    os.chdir(prob_type)

    for run_strat in run_strats[prob_type]:
        if run_strat == 'nocmfd':
            os.system('mkdir -p {}'.format(run_strat))
            os.chdir(run_strat)
            for base_file in cluster_params['base_files']:
                new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                os.system('cp ./../../base/{} ./{}'.format(base_file, new_file))
            create_files(batch_template, cluster_params, prob_type, run_file, run_strat)
            os.chdir('./..')

        else:
            cmfd_mesh = run_strat.split(' ')[0]
            window_size = run_strat.split(' ')[1]
            mesh_dir = 'cmfd-{}-mesh'.format(cmfd_mesh)
            os.system('mkdir -p {}'.format(mesh_dir))
            os.chdir(mesh_dir)
            window_dir = 'expwindow-{}'.format(window_size)
            os.system('mkdir -p {}'.format(window_dir))
            os.chdir(window_dir)
            for base_file in cluster_params['base_files']:
                new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                os.system('cp ./../../../base/{} ./{}'.format(base_file, new_file))
            create_files(batch_template, cluster_params, prob_type, run_file, run_strat)

            os.chdir('./../..')
            
    os.chdir('./..')


def create_files(batch_template, cluster_params, prob_name, run_file, run_strat):
    jobname = '-'.join(run_strat.split(' '))
    nodes = cluster_params['nodes']
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{partition}', cluster_params['partition'])
    batch_template = batch_template.replace('{nproc}', str(nprocs))
    openmc_args = "{} {} {}".format(nthreads, prob_name, run_strat)
    batch_template = batch_template.replace('{openmc_args}', openmc_args)
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    print_str = os.getcwd().split('var_testing/')[-1]
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
