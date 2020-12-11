import sys
import socket
import os

def generate_input_files(cluster, prob_type, run_file, seed_begin, seed_end):
    if cluster == "INL Cluster":
        param_dict = {
            '1d-homog': {
                'batch_file': "job-inl.qsub",
                'run_command': "qsub ",
                'base_files': ['1dh-unif-settings.xml', '1dh-geometry.xml', '1dh-materials.xml', 'run_openmc.py'],
                'ppn': 36,
                'nodes': 4,
                'long_walltime': '96:00:00',
                'walltime': '48:00:00',
                'partition': 'neup'
            },
            '2d-beavrs': {
                'batch_file': "job-inl.qsub",
                'run_command': "qsub ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml', 'run_openmc.py'],
                'ppn': 36,
                'nodes': 4,
                'long_walltime': '96:00:00',
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

    '''
    window_run_strats = {
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
    }'''

    run_strats = {
        '2d-beavrs': [
            'nocmfd 320M',
            'nocmfd 10M',
            'nocmfd 5M',
            'nocmfd 2p5M',
            #'pincell 128 10M'
            #'pincell 128 5M',
            #'pincell 128 2p5M',
            #'pincell 128 1p25M',
            'qassembly 128 320M',
            'qassembly 128 10M',
            'qassembly 128 5M',
            'qassembly 128 2p5M',
            'assembly 128 320M',
            'assembly 128 10M',
            'assembly 128 5M',
            'assembly 128 2p5M',
            'qassembly 64 320M',
            'qassembly 64 10M',
            'qassembly 64 5M',
            'qassembly 64 2p5M',
            'assembly 64 320M',
            'assembly 64 10M',
            'assembly 64 5M',
            'assembly 64 2p5M',
            'qassembly 32 320M',
            'qassembly 32 10M',
            'qassembly 32 5M',
            'qassembly 32 2p5M',
            'assembly 32 320M',
            'assembly 32 10M',
            'assembly 32 5M',
            'assembly 32 2p5M',
            'qassembly 16 320M',
            'qassembly 16 10M',
            'qassembly 16 5M',
            'qassembly 16 2p5M',
            'assembly 16 320M',
            'assembly 16 10M',
            'assembly 16 5M',
            'assembly 16 2p5M',
            'qassembly 256 10M',
            'qassembly 1024 10M',
            'assembly 256 10M',
            'assembly 1024 10M'
        ],
        '1d-homog': [
            'nocmfd 320M',
            'nocmfd 10M',
            'nocmfd 5M',
            'nocmfd 2p5M',
            '0p4cm 128 320M',
            '0p4cm 128 10M',
            '0p4cm 128 5M',
            '0p4cm 128 2p5M',
            '20cm 128 320M',
            '20cm 128 10M',
            '20cm 128 5M',
            '20cm 128 2p5M',
            '0p4cm 64 320M',
            '0p4cm 64 10M',
            '0p4cm 64 5M',
            '0p4cm 64 2p5M',
            '20cm 64 320M',
            '20cm 64 10M',
            '20cm 64 5M',
            '20cm 64 2p5M',
            '0p4cm 32 320M',
            '0p4cm 32 10M',
            '0p4cm 32 5M',
            '0p4cm 32 2p5M',
            '20cm 32 320M',
            '20cm 32 10M',
            '20cm 32 5M',
            '20cm 32 2p5M',
            '0p4cm 16 320M',
            '0p4cm 16 10M',
            '0p4cm 16 5M',
            '0p4cm 16 2p5M',
            '20cm 16 320M',
            '20cm 16 10M',
            '20cm 16 5M',
            '20cm 16 2p5M',
            '0p4cm 256 10M',
            '0p4cm 1024 10M',
            '20cm 256 10M',
            '20cm 1024 10M'
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
        seed_dir = 'seed{}'.format(seed_begin)
        os.system('mkdir -p {}'.format(seed_dir))
        os.chdir(seed_dir)
        for run_strat in run_strats[prob_type]:
            cmfd_mesh = run_strat.split(' ')[0]
            n_particles = run_strat.split(' ')[-1]
            if cmfd_mesh == 'nocmfd':
                run_strat_dir = '{}-{}'.format(cmfd_mesh, n_particles)
                os.system('mkdir -p {}'.format(run_strat_dir))
                os.chdir(run_strat_dir)
                for base_file in cluster_params['base_files']:
                    new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                    os.system('cp ./../../../base/{} ./{}'.format(base_file, new_file))
                int_particles = int(float(n_particles.split('M')[0].replace('p', '.'))*1000000)
                os.system("sed -i 's-10000000-{}-g' settings.xml".format(int_particles))
                os.system("sed -i 's-seed>1-seed>{}-g' settings.xml".format(seed))
                if n_particles == '10M':
                    os.system("sed -i 's-inactive>498-inactive>1998-g' settings.xml")
                    os.system("sed -i 's-batches>499-batches>1999-g' settings.xml")
                create_files(batch_template, cluster_params, prob_type, run_file, run_strat)
                os.chdir('./../')

            else:
                window_size = run_strat.split(' ')[1]
                mesh_dir = 'cmfd-{}-mesh'.format(cmfd_mesh)
                os.system('mkdir -p {}'.format(mesh_dir))
                os.chdir(mesh_dir)
                window_dir = 'expwindow-{}-{}'.format(window_size, n_particles)
                os.system('mkdir -p {}'.format(window_dir))
                os.chdir(window_dir)
                for base_file in cluster_params['base_files']:
                    new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                    os.system('cp ./../../../../base/{} ./{}'.format(base_file, new_file))
                int_particles = int(float(n_particles.split('M')[0].replace('p', '.'))*1000000)
                os.system("sed -i 's-10000000-{}-g' settings.xml".format(int_particles))
                os.system("sed -i 's-seed>1-seed>{}-g' settings.xml".format(seed))
                if window_size == '1024':
                    os.system("sed -i 's-inactive>498-inactive>1998-g' settings.xml")
                    os.system("sed -i 's-batches>499-batches>1999-g' settings.xml")
                create_files(batch_template, cluster_params, prob_type, run_file, run_strat)

                os.chdir('./../../')
        os.chdir('./..')
    os.chdir('./..')


def create_files(batch_template, cluster_params, prob_name, run_file, run_strat):
    if '1024' in run_strat or '320M' in run_strat:
        walltime = cluster_params['long_walltime']
        print('Walltime doubled')
        print(run_strat)
    else:
        walltime = cluster_params['walltime']
    run_strat = ' '.join(run_strat.split(' ')[:-1])
    jobname = '-'.join(run_strat.split(' '))
    nodes = cluster_params['nodes']
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{walltime}', walltime)
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
    if len(sys.argv) not in [4, 5]:
        print('Usage: generate_cmfd_fet_examples.py [prob_type] [seed_begin] [seed_end] [-r]')
        sys.exit()

    # Get command line arguments
    prob_type = sys.argv[1]
    seed_begin = int(sys.argv[2])
    seed_end = int(sys.argv[3])
    run_file = len(sys.argv) == 5 and sys.argv[4] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, prob_type, run_file, seed_begin, seed_end)
