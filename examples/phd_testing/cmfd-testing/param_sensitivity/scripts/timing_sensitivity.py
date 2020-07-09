import sys
import socket
import os

def generate_input_files(cluster, trial_begin, trial_end, prob_type, run_file):
    if cluster == "Green Cluster":
        param_dict = {
            '1d-homog': {
                'batch_file': "job.slurm",
                'run_command': "sbatch ",
                'base_files': ['1dh-unif-settings.xml', '1dh-geometry.xml', '1dh-materials.xml'],
                'ppn': 32,
                'nodes': 1,
                'walltime': '12:00:00'
            },
            '2d-beavrs': {
                'batch_file': "job.slurm",
                'run_command': "sbatch ",
                'base_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml'],
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
        '1d-homog': [
            'nocmfd noea',
            'nocmfd 0p4cm-ent',
            'nocmfd 20cm-ent',
            'nocmfd fet',
            '0p4cm noea',
            '0p4cm 0p4cm-ent',
            '0p4cm 20cm-ent',
            '0p4cm fet',
            '20cm noea',
            '20cm 0p4cm-ent',
            '20cm 20cm-ent',
            '20cm fet'
        ],
        '2d-beavrs': [
            'nocmfd noea',
            'nocmfd pincell-ent',
            'nocmfd qassem-ent',
            'nocmfd fet',
            'assembly noea',
            'assembly pincell-ent',
            'assembly qassem-ent',
            'assembly fet',
            'qassembly noea',
            'qassembly pincell-ent',
            'qassembly qassem-ent',
            'qassembly fet',
            'pincell noea',
            'pincell pincell-ent',
            'pincell qassem-ent',
            'pincell fet'
        ]
    }

    sed_commands = {
        'fet': [
            "sed -i 's~<mesh~<!--mesh~g' settings.xml",
            "sed -i 's~</entropy_mesh>~</entropy_mesh-->~g' settings.xml"
        ],
        'noea': [
            "sed -i 's~<mesh~<!--mesh~g' settings.xml",
            "sed -i 's~</entropy_mesh>~</entropy_mesh-->~g' settings.xml",
            "sed -i 's~<fet_convergence>~<!--fet_convergence>~g' settings.xml",
            "sed -i 's~</fet_convergence>~</fet_convergence-->~g' settings.xml"
        ],
        '0p4cm-ent': [
            "sed -i 's~<fet_convergence>~<!--fet_convergence>~g' settings.xml",
            "sed -i 's~</fet_convergence>~</fet_convergence-->~g' settings.xml"
        ],
        '20cm-ent': [
            "sed -i 's~<fet_convergence>~<!--fet_convergence>~g' settings.xml",
            "sed -i 's~</fet_convergence>~</fet_convergence-->~g' settings.xml",
            "sed -i 's~1 1 1000~1 1 20~g' settings.xml"
        ],
        'pincell-ent': [
            "sed -i 's~<fet_convergence>~<!--fet_convergence>~g' settings.xml",
            "sed -i 's~</fet_convergence>~</fet_convergence-->~g' settings.xml",
            "sed -i 's~34 34 1~289 289 1~g' settings.xml"
        ],
        'qassem-ent': [
            "sed -i 's~<fet_convergence>~<!--fet_convergence>~g' settings.xml",
            "sed -i 's~</fet_convergence>~</fet_convergence-->~g' settings.xml"
        ],
    }

    os.system('mkdir -p ./../timing-sensitivity')
    os.chdir('./../timing-sensitivity')

    # Generate file templates
    with open('./../base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    # Generate files for each problem type
    os.system('mkdir -p {}'.format(prob_type))
    os.chdir(prob_type)

    for trial in range(trial_begin, trial_end+1):
        trial_dir = 'trial{}'.format(str(trial))
        os.system('mkdir -p {}'.format(trial_dir))
        os.chdir(trial_dir)

        for run_strat in run_strats[prob_type]:
            cmfd_mesh, test_name = run_strat.split(' ')
            mesh_dir = cmfd_mesh if cmfd_mesh == 'nocmfd' else 'cmfd-{}-mesh'.format(cmfd_mesh)
            os.system('mkdir -p {}'.format(mesh_dir))
            os.chdir(mesh_dir)

            test_dir = '{}node-{}'.format(cluster_params['nodes'], test_name)
            os.system('mkdir -p {}'.format(test_dir))
            os.chdir(test_dir)

            # Copy base files to directory
            for base_file in cluster_params['base_files']:
                new_file = base_file.split('-')[-1] if 'xml' in base_file else base_file
                os.system('cp ./../../../../../base/{} ./{}'.format(base_file, new_file))
            os.system('cp ./../../../../../base/run_openmc_timing_sensitivity.py run_openmc.py')
            # Run sed commands specific to test type
            for sed_command in sed_commands[test_name]:
                os.system(sed_command)

            # Set number of batches and inactives
            n_batches = 10 if cmfd_mesh == 'pincell' else 100
            os.system("sed -i 's~<batches>.*</batches>~<batches>{}</batches>~g' settings.xml".format(n_batches))
            os.system("sed -i 's~<inactive>.*</inactive>~<inactive>{}</inactive>~g' settings.xml".format(n_batches - 1))
            os.system("sed -i 's~<particles>.*</particles>~<particles>10000000</particles>~g' settings.xml")

            # Create and run batch file
            create_files(batch_template, cluster_params, trial, prob_type, run_file, run_strat)

            os.chdir('./../..')
        os.chdir('./..')
    os.chdir('./..')


def create_files(batch_template, cluster_params, trial, prob_name, run_file, run_strat):
    jobname = 't{}-{}'.format(trial, '-'.join(run_strat.split(' ')))
    nodes = cluster_params['nodes']
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{tasks}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{nprocs}', str(nprocs))
    batch_template = batch_template.replace('{nthreads}', str(nthreads))
    openmc_args = '{} {}'.format(prob_name, run_strat.split(' ')[0])
    batch_template = batch_template.replace('{prob_type}', openmc_args)
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    print_str = os.getcwd().split('timing-sensitivity/')[-1]
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
        print('Usage: timing_sensitivity.py [trial_begin] [trial_end] [prob_type] [-r]')
        sys.exit()

    # Get command line arguments
    trial_begin = int(sys.argv[1])
    trial_end = int(sys.argv[2])
    prob_type = sys.argv[3]
    run_file = len(sys.argv) == 5 and sys.argv[4] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, trial_begin, trial_end, prob_type, run_file)
