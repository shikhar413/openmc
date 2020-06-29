import sys
import socket
import os

def generate_input_files(cluster, run_file):
    if cluster == "Green Cluster":
        cluster_params = {
            'prob_type': "2d-beavrs",
            'particles': 10000000,
            'mesh_types': ['assembly', 'qassembly', 'pincell', 'none'],
            'batch_file': "job.slurm",
            'run_command': "sbatch ",
            'xml_files': ['2db-settings.xml', '2db-geometry.xml', '2db-materials.xml'],
            'ppn': 32,
            'nodes': [2],
            'walltime': '12:00:00',
            'cmfd_threads': [1,2,4,8,16],
            'ref_d': '[1.42669, 0.400433]',
            'n_batches': 100,
            'n_inactive': 99
        }

    os.system('mkdir -p ./../thread-sensitivity')
    os.chdir('./../thread-sensitivity')

    # Generate files for base run
    with open('../base/params.cfg', 'r') as file:
        param_template = file.read()
    with open('../base/{}'.format(cluster_params['batch_file']), 'r') as file:
        batch_template = file.read()

    os.system('mkdir -p {}'.format(cluster_params['prob_type']))
    os.chdir(cluster_params['prob_type'])

    # Create files for cmfd case
    for mesh in cluster_params['mesh_types']:
        if mesh == 'none':
            mesh_dir = 'nocmfd'
        else:
            mesh_dir = 'cmfd-{}-mesh'.format(mesh)
        os.system('mkdir -p {}'.format(mesh_dir))
        os.chdir(mesh_dir)
        nodes = cluster_params['nodes']
        if mesh == 'pincell':
            nodes = [2, 11]
        if mesh == 'none':
            for xml_file in cluster_params['xml_files']:
                os.system('cp ./../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))
            thread = None
            node = 1
            create_files(param_template, batch_template, cluster_params, node, thread, run_file, mesh)
            os.system('sed -i "s/python3 .*/openmc -s 16/g" job.slurm')
        else:
            for node in nodes:
                for thread in cluster_params['cmfd_threads']:
                    nodethread_dir = '{}node{}thread'.format(node-1, thread)
                    os.system('mkdir -p {}'.format(nodethread_dir))
                    os.chdir(nodethread_dir)
                    for xml_file in cluster_params['xml_files']:
                        os.system('cp ./../../../../base/{} ./{}'.format(xml_file, xml_file.split('-')[-1]))
                    os.system('cp ./../../../../base/run_ea_cmfd.py ./run_openmc.py')
                    create_files(param_template, batch_template, cluster_params, node, thread, run_file, mesh)
                    os.chdir('./..')
        os.chdir('./..')
    os.chdir('./..')

    # TODO create test cases for 1d-homog case (0.4cm mesh vs. 20cm mesh)
    # TODO add cases for when n_threads is fixed and add entropy, fet calculation


def create_files(param_template, batch_template, cluster_params, node, thread, run_file, mesh):
    os.system('sed -i s@\</fet_convergence@\</fet_convergence--@g settings.xml')
    os.system('sed -i s@\<fet@\<\!--fet@g settings.xml')
    os.system('sed -i s@\<mesh@\<\!--mesh@g settings.xml')
    os.system('sed -i s@\</entropy_mesh@\</entropy_mesh--@g settings.xml')
    if mesh == 'none':
        jobname = 'nocmfd'
    else:
        jobname = 'nth{}{}'.format(thread, mesh)
    nodes = node
    tasks = nodes * cluster_params['ppn']
    nprocs = nodes*2
    nthreads = int(cluster_params['ppn']/2)

    batch_template = batch_template.replace('{prob_type}', mesh)
    batch_template = batch_template.replace('{tasks}', str(tasks))
    batch_template = batch_template.replace('{nodes}', str(nodes))
    batch_template = batch_template.replace('{ppn}', str(cluster_params['ppn']))
    batch_template = batch_template.replace('{job_name}', jobname)
    batch_template = batch_template.replace('{walltime}', cluster_params['walltime'])
    batch_template = batch_template.replace('{nprocs}', str(nprocs))
    batch_template = batch_template.replace('{nthreads}', cluster_params['prob_type'])
    with open(cluster_params['batch_file'], 'w') as f:
        f.write(batch_template)

    if mesh != 'none':
        param_template = param_template.replace('{ref_d}', cluster_params['ref_d'])
        param_template = param_template.replace('{n_threads}', str(thread))
        n_batches = str(cluster_params['n_batches'])
        n_inactive = str(cluster_params['n_inactive'])
        if mesh == 'pincell' and node == 2:
            n_batches = '10'
            n_inactive = '9'
        param_template = param_template.replace('{n_batches}', n_batches)
        param_template = param_template.replace('{n_inactive}', n_inactive)

        with open('params.cfg', 'w') as f:
            f.write(param_template)

    print_str = os.getcwd().split('thread-sensitivity/')[-1]
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
    if len(sys.argv) not in [1, 2]:
        print('Usage: tally_solver_begin_sensitivity.py [-r]')
        sys.exit()

    # Get command line arguments
    run_file = len(sys.argv) == 2 and sys.argv[1] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(cluster, run_file)