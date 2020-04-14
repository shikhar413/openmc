import sys
import socket
import os

def generate_input_files(cluster, n_seeds, run_file):
    cmfd_probs = ['expwindow-16limit-activeon', 'expwindow-64limit-activeon',
                  'fixwindow-16limit-activeon', 'expwindow-64limit-activeoff',
                  'expwindow-nolimit-activeon', 'fixwindow-64limit-activeon']
    df_cmfd_probs = ['expwindow-16limit-activeon', 'expwindow-64limit-activeon',
                  'fixwindow-16limit-activeon', 'fixwindow-64limit-activeon']
    if cluster == "Green Cluster":
        cluster_params = {
            'prob_type': "1d-homog",
            'particles': [100000, 1000000, 10000000],
            'mesh_types': ['4cm', '1cm', '0p1cm'],
            'batch_file': "job.slurm",
            'run_command': "sbatch ",
            'xml_files': ['1dh-unif-settings.xml', '1dh-geometry.xml', '1dh-materials.xml'],
            'ppn': 32,
            'nodes': 1,
            'walltime': '12:00:00',
            'solver_end': 150
        }
    elif cluster == "INL Cluster":
        cluster_params = {
            'prob_type': "2d-beavrs",
            'particles': [10000000, 100000000],
            'mesh_types': ['assembly', 'qassembly'],
            'batch_file': "job-inl.qsub",
            'run_command': "qsub ",
            'xml_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml'],
            'ppn': 36,
            'nodes': 1,
            'walltime': '12:00:00',
            'solver_end': 200
        }

    os.system('mkdir -p ./../mesh-window-sensitivity')
    os.chdir('./../mesh-window-sensitivity')

    # Generate files for base run
    with open('../base/run_openmc_cmfd_mesh_window.py', 'r') as file:
        cmfd_template = file.read()
    with open('../base/run_openmc_mesh_window.py', 'r') as file:
        nocmfd_template = file.read()
    with open('../base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    os.system('mkdir -p {}'.format(cluster_params['prob_type']))
    os.chdir(cluster_params['prob_type'])

    for seed in n_seeds:
        seed_dir = 'seed{}'.format(str(seed))
        os.system('mkdir -p {}'.format(seed_dir))
        os.chdir(seed_dir)

        for particle in cluster_params['particles']:
            particle_dir = str(particle) + "n"
            os.system('mkdir -p {}'.format(particle_dir))
            os.chdir(particle_dir)

            # Create files for nocmfd case
            os.system('mkdir -p nocmfd')
            os.chdir('nocmfd')
            for xml_file in cluster_params['xml_files']:
                os.system('cp ./../../../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))
            create_files(nocmfd_template, batch_template, particle, cluster_params, seed, 'nocmfd', run_file)
            os.chdir('./..')
            # Create files for cmfd case
            for mesh in cluster_params['mesh_types']:
                mesh_dir = 'cmfd-{}-mesh'.format(mesh)
                os.system('mkdir -p {}'.format(mesh_dir))
                os.chdir(mesh_dir)
                for cmfd_prob in cmfd_probs:
                    os.system('mkdir -p {}'.format(cmfd_prob))
                    os.chdir(cmfd_prob)
                    for xml_file in cluster_params['xml_files']:
                        os.system('cp ./../../../../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))
                    create_files(cmfd_template, batch_template, particle, cluster_params, seed, cmfd_prob, run_file, mesh=mesh)
                    os.chdir('./..')
                os.chdir('./..')
            # Create files for df-cmfd case
            for mesh in cluster_params['mesh_types']:
                mesh_dir = 'df-cmfd-{}-mesh'.format(mesh)
                os.system('mkdir -p {}'.format(mesh_dir))
                os.chdir(mesh_dir)
                for cmfd_prob in df_cmfd_probs:
                    os.system('mkdir -p {}'.format(cmfd_prob))
                    os.chdir(cmfd_prob)
                    for xml_file in cluster_params['xml_files']:
                        os.system('cp ./../../../../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))
                    create_files(cmfd_template, batch_template, particle, cluster_params, seed, cmfd_prob, run_file, mesh=mesh, df=True)
                    os.chdir('./..')
                os.chdir('./..')
            os.chdir('./..')
        os.chdir('./..')
    os.chdir('./..')


def create_files(py_template, batch_template, nparticles, cluster_params, seed, prob_name, run_file, mesh=None, df=False):
    all_prob_params = {
        'nocmfd': {
            'solver_end': '',
            'window_type': '',
            'window_size': '',
            'max_window_size': ''
        },
        'expwindow-16limit-activeon': {
            'solver_end': '',
            'window_type': 'expanding',
            'window_size': '',
            'max_window_size': 'cmfd_run.max_window_size = 16'
        },
        'expwindow-64limit-activeon': {
            'solver_end': '',
            'window_type': 'expanding',
            'window_size': '',
            'max_window_size': 'cmfd_run.max_window_size = 64'
        },
        'fixwindow-16limit-activeon': {
            'solver_end': '',
            'window_type': 'rolling',
            'window_size': 'cmfd_run.window_size = 16',
            'max_window_size': ''
        },
        'expwindow-64limit-activeoff': {
            'solver_end': 'cmfd_run.solver_end = {}'.format(cluster_params['solver_end']),
            'window_type': 'expanding',
            'window_size': '',
            'max_window_size': 'cmfd_run.max_window_size = 64'
        },
        'expwindow-nolimit-activeon': {
            'solver_end': '',
            'window_type': 'expanding',
            'window_size': '',
            'max_window_size': ''
        },
        'fixwindow-64limit-activeon': {
            'solver_end': '',
            'window_type': 'rolling',
            'window_size': 'cmfd_run.window_size = 64',
            'max_window_size': ''
        }
    }

    os.system("sed -i 's-<particles>1000000</particles>-<particles>{}</particles>-g' settings.xml".format(nparticles))
    os.system("sed -i 's-<seed>1</seed>-<seed>{}</seed>-g' settings.xml".format(seed))
    jobname = 's{}{}'.format(seed,prob_name)
    nodes = cluster_params['nodes']
    if nparticles > 10000000:
        nodes *= 10
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{prob_type}', cluster_params['prob_type'])
    batch_template = batch_template.replace('{tasks}', str(tasks))
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{nprocs}', str(nprocs))
    batch_template = batch_template.replace('{nthreads}', str(nthreads))
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    prob_params = all_prob_params[prob_name]
    if df:
        prob_params['max_window_size'] += "\n    cmfd_run.damping_factor = 0.5"

    py_template = py_template.replace('{solver_end}', prob_params['solver_end'])
    py_template = py_template.replace('{window_type}', prob_params['window_type'])
    py_template = py_template.replace('{window_size}', prob_params['window_size'])
    py_template = py_template.replace('{max_window_size}', prob_params['max_window_size'])

    if mesh is not None:
        dims, map_str = get_mesh_strings(mesh, 'cm' in mesh)
        py_template = py_template.replace('{cmfd_dim}', dims)
        py_template = py_template.replace('{map_str}', map_str)
    with open('run_openmc.py', 'w') as f:
        f.write(py_template)

    print_str = os.getcwd().split('mesh-window-sensitivity/')[-1]
    if run_file:
        print('Running batch script in {}'.format(print_str))
        os.system(cluster_params['run_command']+cluster_params['batch_file'])
    else:
        print('Created input files in {}'.format(print_str))


def get_mesh_strings(mesh_type, is_1d):
    mesh_dims = {
        '4cm': '1, 1, 100',
        '1cm': '1, 1, 400',
        '0p1cm': '1, 1, 4000',
        'assembly': '17, 17, 1',
        'qassembly': '34, 34, 1'
}
    map_str = {
        'assembly': '''[
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]''',
        'qassembly': '''[
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]'''
}
    if is_1d:
        return mesh_dims[mesh_type], ''
    else:
        return mesh_dims[mesh_type], 'cmfd_mesh.map = ' + map_str[mesh_type]

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
