import sys
import socket
import os

import openmc


def build_homog_input_files(order, source_loc, bc, prob_size,
                                  num_neutrons, num_batches, SE_dim,
                                  seed, is_3d, dir):
    os.system("mkdir -p {dir}".format(dir=dir))
    os.chdir(dir)

    xmin = -1.*prob_size[0]/2.
    xmax = prob_size[0]/2.
    ymin = -1.*prob_size[1]/2.
    ymax = prob_size[1]/2.
    zmin = -1.*prob_size[2]/2.
    zmax = prob_size[2]/2.

    # Instantiate some Materials and register the appropriate Nuclides
    uranyl_sulf = openmc.Material(name='Uranyl Sulfate')
    uranyl_sulf.set_density("atom/b-cm", 9.9035E-02)
    uranyl_sulf.add_nuclide("U235", 9.6795E-05)
    uranyl_sulf.add_nuclide("U234", 7.4257E-07)
    uranyl_sulf.add_nuclide("U238", 5.5518E-04)
    uranyl_sulf.add_nuclide("S32" , 6.5272E-04)
    uranyl_sulf.add_nuclide("O16" , 3.5185E-02)
    uranyl_sulf.add_nuclide("H1"  , 6.2538E-02)
    uranyl_sulf.add_s_alpha_beta('c_H_in_H2O')
    uranyl_sulf.depletable = False

    # Instantiate a Materials collection and export to XML
    materials_file = openmc.Materials([uranyl_sulf])
    materials_file.export_to_xml()

    # Instantiate planar surfaces
    x1 = openmc.XPlane(x0=xmin)
    x2 = openmc.XPlane(x0=xmax)
    y1 = openmc.YPlane(y0=ymin)
    y2 = openmc.YPlane(y0=ymax)
    z1 = openmc.ZPlane(z0=zmin)
    z2 = openmc.ZPlane(z0=zmax)

    # Set boundary conditions
    surface_list = [x1, x2, y1, y2, z1, z2]
    for i in range(len(surface_list)):
        surface = surface_list[i]
        surface.boundary_type = bc[i]

    # Define Cell, Region, and Fill
    uran_sulf_sol = openmc.Cell(name='uranyl sulfate solution')
    uran_sulf_sol.region = +x1 & -x2 & +y1 & -y2 & +z1 & -z2
    uran_sulf_sol.fill = uranyl_sulf

    # Instantiate root universe
    root = openmc.Universe(name='root universe')
    root.add_cells([uran_sulf_sol])

    # Instantiate a Geometry, register the root Universe, and export to XML
    geometry = openmc.Geometry(root)
    geometry.export_to_xml()

    # Define runtime settings
    settings = openmc.Settings()
    point_source = openmc.stats.Point(xyz=source_loc)
    settings.source = openmc.Source(space=point_source)
    settings.batches = num_batches
    settings.inactive = num_batches - 1
    settings.particles = num_neutrons

    # CHANGE
    entropy_mesh = openmc.Mesh()
    entropy_mesh.lower_left = [xmin, ymin, zmin]
    entropy_mesh.upper_right = [xmax, ymax, zmax]
    entropy_mesh.dimension = SE_dim
    settings.entropy_mesh = entropy_mesh
    settings.seed = seed
    settings.export_to_xml()

    # Create a nu-fission tally
    nu_fiss_tally = openmc.Tally()
    nu_fiss_tally.scores = ['nu-fission']

    # Create a Legendre polynomial expansion filter and add to tally
    if not is_3d:
        expand_filter = openmc.SpatialLegendreFilter(order, 'z', zmin, zmax)
        nu_fiss_tally.filters.append(expand_filter)
    else:
        expand_filter_x = openmc.SpatialLegendreFilter(order, 'x', xmin, xmax)
        expand_filter_y = openmc.SpatialLegendreFilter(order, 'y', ymin, ymax)
        expand_filter_z = openmc.SpatialLegendreFilter(order, 'z', zmin, zmax)
        nu_fiss_tally.filters.append(expand_filter_x)
        nu_fiss_tally.filters.append(expand_filter_y)
        nu_fiss_tally.filters.append(expand_filter_z)

    tallies = openmc.Tallies([nu_fiss_tally])
    tallies.export_to_xml()

    os.chdir("./..")


def get_source_locs(prob_type, prob_size = None):
    source_loc_dict = {
        '1d-homog': {
            100.: [(0.0, 0.0, 0.0), (0.0, 0.0, 40.0)],
            200.: [(0.0, 0.0, 0.0), (0.0, 0.0, 90.0)],
            400.: [(0.0, 0.0, 0.0), (0.0, 0.0, 190.0)],
            600.: [(0.0, 0.0, 0.0), (0.0, 0.0, 290.0)],
            800.: [(0.0, 0.0, 0.0), (0.0, 0.0, 390.0)]
        },
        '2d-beavrs':
            [(-161.2773, -161.2773, 220.0, 161.2773, 161.2773, 230.0),
             (0.0, -161.2773, 220.0, 161.2773, 161.2773, 230.0)],
        '3d-exasmr':
            [(-75.26274, -75.26274, 36.007, 75.26274, 75.26274, 236.0066),
             (0.0, -75.26274, 36.007, 75.26274, 75.26274, 236.0066)],
        '3d-homog':
            [(0.,0.,0.)]
    }

    if prob_size is None:
        return source_loc_dict[prob_type]
    else:
        return source_loc_dict[prob_type][prob_size[2]]


def get_num_batches(prob_type, prob_size = None):
    num_batch_dict = {
        '1d-homog': {
            100.: [399, 499],
            200.: [499, 499],
            400.: [599, 999],
            600.: [999, 2099],
            800.: [1499, 2999]
        },
        '2d-beavrs': [499, 999],
        '3d-exasmr': [149, 199],
        '3d-homog': [499]
    }

    if prob_size is None:
        return num_batch_dict[prob_type]
    else:
        return num_batch_dict[prob_type][prob_size[2]]


def get_num_nodes(cluster, prob_type, num_neutrons):
    # TODO Define method as dictionary by getting rid of special case for SK Linux
    if cluster == 'SK Linux':
        return 1
    else:
        num_node_dict = {
            '1d-homog': {
                10000: 5,
                100000: 5,
                1000000: 5,
                10000000: 20
            },
            '2d-beavrs': {
                100000: 10,
                1000000: 10,
                4000000: 10,
                10000000: 15,
                40000000: 30
            },
            '3d-exasmr': {
                100000: 5,
                1000000: 15,
                10000000: 30,
                40000000: 30
            },
            '3d-homog': {
                10000: 5,
                100000: 10,
                1000000: 30
            }
        }
        return num_node_dict[prob_type][num_neutrons]


def get_script_name(prob_type, num_neutrons, seed, is_offset, size=None):
    short_prob_type = prob_type[0:2] + prob_type[3]
    short_num_neutron_dict = {
        10000: '10K',
        100000: '100K',
        1000000: '1M',
        4000000: '4M',
        10000000: '10M',
        40000000: '40M',
    }
    short_num_neutron = short_num_neutron_dict[num_neutrons]
    if size is not None:
        script_name = '{}{}{}s{}cm'.format(short_prob_type, short_num_neutron,
                                           str(seed), str(size))
    else:
        script_name = '{}{}{}s'.format(short_prob_type, short_num_neutron,
                                       str(seed))
    if is_offset:
        script_name += 'o'

    return script_name


def create_file_from_template(replace_dict, template_path, outfile, dir):
    with open(template_path, 'r') as file:
      new_file=file.read()
    for key in replace_dict:
        new_file = new_file.replace(key, str(replace_dict[key]))
    with open(outfile, "w") as file:
      file.write(new_file)
    os.system("mv {} {}".format(outfile, dir))


def build_pbs_batch_script(ppn, spn, num_nodes, name, dir, run_file,
                           prob_type, is_inl):
    num_mpi_procs = num_nodes * spn
    num_omp_threads = int(ppn / spn)

    replace_dict = {
        '{procs_per_node}': str(ppn),
        '{num_nodes}': str(num_nodes),
        '{filename}': str(name),
        '{num_mpi_procs}': num_mpi_procs,
        '{num_omp_threads}': str(num_omp_threads),
        '{prob_type}': str(prob_type)
    }
    if is_inl:
        template_path = '../../scripts/inl_batch_script_template.qsub'
    else:
        template_path = '../../scripts/nse_batch_script_template.qsub'
    outfile = name + '.pbs'
    create_file_from_template(replace_dict, template_path, outfile, dir)

    if run_file:
        os.chdir(dir)
        # TODO Add some check to see if run command should be issued
        print("Running job in directory {}".format(dir))
        os.system('qsub ' + outfile)
        os.chdir("./..")


def build_slurm_batch_script(ppn, spn, num_nodes, name, dir, run_file,
                             prob_type):
    num_procs = num_nodes * ppn
    num_mpi_procs = num_nodes * spn
    num_omp_threads = int(ppn / spn)

    replace_dict = {
        '{num_procs}': str(ppn*num_nodes),
        '{num_nodes}': str(num_nodes),
        '{filename}': str(name),
        '{num_mpi_procs}': num_mpi_procs,
        '{num_omp_threads}': str(num_omp_threads),
        '{prob_type}': str(prob_type)
    }
    template_path = '../../scripts/batch_script_template.slurm'
    outfile = name + '.slurm'
    create_file_from_template(replace_dict, template_path, outfile, dir)

    if run_file:
        os.chdir(dir)
        # TODO Add some check to see if run command should be issued
        print("Running job in directory {}".format(dir))
        os.system('sbatch ' + outfile)
        os.chdir("./..")

def build_batch_script_files(cluster, prob_type, num_neutrons, name, dir,
                             run_file):
    cluster_param_dict = {
        'SK Linux': {'ppn': 8, 'spn': 1, 'cluster_type': ['pbs', 'slurm']},
        'NSE Cluster': {'ppn': 12, 'spn': 2, 'cluster_type': ['pbs']},
        'Green Cluster': {'ppn': 32, 'spn': 2, 'cluster_type': ['slurm']},
        'INL Cluster': {'ppn': 36, 'spn': 2, 'cluster_type': ['pbs']}
    }
    cluster_type = cluster_param_dict[cluster]['cluster_type']
    ppn = cluster_param_dict[cluster]['ppn']
    spn = cluster_param_dict[cluster]['spn']
    num_nodes = get_num_nodes(cluster, prob_type, num_neutrons)

    if 'pbs' in cluster_type:
        is_inl = cluster == 'INL Cluster'
        build_pbs_batch_script(ppn, spn, num_nodes, name, dir, run_file,
                               prob_type, is_inl)
    if 'slurm' in cluster_type:
        build_slurm_batch_script(ppn, spn, num_nodes, name, dir, run_file,
                                 prob_type)
    os.system('cp ../../scripts/run_openmc_capi.py {}'.format(dir))


def build_benchmark_input_files(prob_type, params, dir, num_batch,
                                num_neutrons, seed, source_loc):

    os.system("mkdir -p {dir}".format(dir=dir))

    geometry_replace_dict = {}
    template_path = '../base/geometry_template.xml'
    outfile = 'geometry.xml'
    create_file_from_template(geometry_replace_dict, template_path, outfile,
                              dir)

    materials_replace_dict = {}
    template_path = '../base/materials_template.xml'
    outfile = 'materials.xml'
    create_file_from_template(materials_replace_dict, template_path, outfile,
                              dir)

    SE_dim = params['SE_dim']
    settings_replace_dict = {
        '{num_particles}': str(num_neutrons),
        '{num_batches}': str(num_batch),
        '{num_inactive}': str(num_batch - 1),
        '{source_loc}': ' '.join(str(s) for s in source_loc),
        '{SE_dim}': ' '.join(str(s) for s in SE_dim),
        '{seed}': str(seed)
    }
    template_path = '../base/settings_template.xml'
    outfile = 'settings.xml'
    create_file_from_template(settings_replace_dict, template_path, outfile,
                              dir)

    if prob_type == '2d-beavrs':
        tallies_replace_dict = {
            '{tally_order}': str(params['tally_order'])
        }
        template_path = '../base/tallies_template.xml'
        outfile = 'tallies.xml'
        create_file_from_template(tallies_replace_dict, template_path, outfile,
                                  dir)
    elif prob_type == '3d-exasmr':
        tallies_replace_dict = {
            '{zern_tally_order}': str(params['zern_tally_order']),
            '{leg_tally_order}': str(params['leg_tally_order']),
        }
        template_path = '../base/tallies_template.xml'
        outfile = 'tallies.xml'
        create_file_from_template(tallies_replace_dict, template_path, outfile,
                                  dir)


def generate_input_files(prob_type, seed, cluster, run_file):
    #TODO get rid of cluster as an input parameter, pass in params
    # Get problem parameters
    params = get_problem_params(prob_type, cluster)

    if prob_type in ['1d-homog', '3d-homog']:
        is_3d = prob_type == '3d-homog'
        target_dir = '../{}/seed{}'.format(prob_type, seed)
        os.system('mkdir -p {}'.format(target_dir))
        os.chdir(target_dir)

        for num_neutrons in params['num_neutrons']:
            order = params['tally_order']
            bc = params['bc']
            SE_dim = params['SE_dim']
            for prob_size in params['prob_sizes']:
                if not is_3d:
                    source_locs = get_source_locs(prob_type,
                                                  prob_size=prob_size)
                    num_batches = get_num_batches(prob_type,
                                                  prob_size=prob_size)
                else:
                    source_locs = get_source_locs(prob_type)
                    num_batches = get_num_batches(prob_type)
                for i in range(len(source_locs)):
                    source_loc = source_locs[i]
                    num_batch = num_batches[i]
                    dir = "{neutrons}n-{size}cm-{loc}source" \
                        .format(loc=source_loc[2], neutrons=num_neutrons,
                                size=prob_size[2])
                    build_homog_input_files(order, source_loc, bc,
                                            prob_size, num_neutrons,
                                            num_batch, SE_dim, seed,
                                            is_3d, dir)
                    script_name = get_script_name(prob_type, num_neutrons, seed, i,
                                                  size=int(prob_size[2]))
                    build_batch_script_files(cluster, prob_type, num_neutrons,
                                             script_name, dir, run_file)
    else:
        target_dir = '../{}/seed{}'.format(prob_type, seed)
        os.system('mkdir -p {}'.format(target_dir))
        os.chdir(target_dir)

        for num_neutrons in params['num_neutrons']:
            source_locs = get_source_locs(prob_type)
            num_batches = get_num_batches(prob_type)
            for i in range(len(source_locs)):
                source_loc = source_locs[i]
                num_batch = num_batches[i]
                dir = "{neutrons}n".format(neutrons=num_neutrons)
                if i == 1:
                    dir += "-offset"
                build_benchmark_input_files(prob_type, params, dir, num_batch,
                                            num_neutrons, seed, source_loc)
                script_name = get_script_name(prob_type, num_neutrons, seed, i)
                build_batch_script_files(cluster, prob_type, num_neutrons,
                                         script_name, dir, run_file)


def get_problem_params(prob_type, cluster):
    problem_param_dict = {
        'SK Linux': {
            '1d-homog': {
                'tally_order': 30,
                'num_neutrons': [10000, 100000, 1000000, 10000000],
                'SE_dim': [1, 1, 1000],
                'bc': ['reflective', 'reflective', 'reflective', 'reflective',
                       'vacuum', 'vacuum'],
                'prob_sizes': [(10., 10., 100.), (10., 10., 200.),
                               (10., 10., 400.), (10., 10., 600.),
                               (10., 10., 800.)]
            },
            '2d-beavrs': {
                'tally_order': 20,
                'num_neutrons': [100000, 1000000, 4000000, 10000000, 40000000],
                'SE_dim': [68, 68, 1]
            },
            '3d-exasmr': {
                'leg_tally_order': 30,
                'zern_tally_order': 20,
                'num_neutrons': [100000, 1000000, 10000000, 40000000],
                'SE_dim': [28, 28, 20]
            },
            '3d-homog': {
                'tally_order': 30,
                'num_neutrons': [10000, 100000, 1000000],
                'SE_dim': [16, 16, 16],
                'bc': ['reflective', 'reflective', 'reflective', 'reflective',
                       'reflective', 'reflective'],
                'prob_sizes': [(100., 100., 100.), (200., 200., 200.),
                               (400., 400., 400.)]
            }
        },
        'NSE Cluster': {
            '1d-homog': {
                'tally_order': 30,
                'num_neutrons': [10000, 100000, 1000000],
                'SE_dim': [1, 1, 1000],
                'bc': ['reflective', 'reflective', 'reflective', 'reflective',
                       'vacuum', 'vacuum'],
                'prob_sizes': [(10., 10., 100.), (10., 10., 200.),
                               (10., 10., 400.), (10., 10., 600.),
                               (10., 10., 800.)]
            }
        },
        'Green Cluster': {
            '2d-beavrs': {
                'tally_order': 20,
                'num_neutrons': [100000, 1000000, 4000000],
                'SE_dim': [68, 68, 1]
            },
            '3d-exasmr': {
                'leg_tally_order': 30,
                'zern_tally_order': 20,
                'num_neutrons': [100000],
                'SE_dim': [28, 28, 20]
            }
        },
        'INL Cluster': {
            '1d-homog': {
                'tally_order': 30,
                'num_neutrons': [10000000],
                'SE_dim': [1, 1, 1000],
                'bc': ['reflective', 'reflective', 'reflective', 'reflective',
                       'vacuum', 'vacuum'],
                'prob_sizes': [(10., 10., 100.), (10., 10., 200.),
                               (10., 10., 400.), (10., 10., 600.),
                               (10., 10., 800.)]
            },
            '2d-beavrs': {
                'tally_order': 20,
                'num_neutrons': [10000000, 40000000],
                'SE_dim': [68, 68, 1]
            },
            '3d-exasmr': {
                'leg_tally_order': 30,
                'zern_tally_order': 20,
                'num_neutrons': [1000000, 10000000, 40000000],
                'SE_dim': [28, 28, 20]
            },
            '3d-homog': {
                'tally_order': 30,
                'num_neutrons': [10000, 10000, 1000000],
                'SE_dim': [16, 16, 16],
                'bc': ['reflective', 'reflective', 'reflective', 'reflective',
                       'reflective', 'reflective'],
                'prob_sizes': [(100., 100., 100.), (200., 200., 200.),
                               (400., 400., 400.)]
            }
        }
    }

    if cluster not in problem_param_dict:
        print('Unexpected cluster {}. Try again.'.format(cluster))
        sys.exit()

    if prob_type not in problem_param_dict[cluster]:
        print('Unexpected problem type {}. Try again.'.format(prob_type))
        sys.exit()

    return problem_param_dict[cluster][prob_type]


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
    # TODO parse arguments more elegantly
    if len(sys.argv) not in [3, 4]:
        print('Usage: generate_examples.py [problem_type] [seed #] [-r]')
        sys.exit()

    # Get command line arguments
    prob_type = sys.argv[1]
    try:
        seed = int(sys.argv[2])
    except ValueError:
        print('Seed number must be of type int')
        sys.exit()
    run_file = len(sys.argv) == 4 and sys.argv[3] == '-r'

    # Get cluster where script is running on
    cluster = get_cluster(socket.gethostname())

    # Generate OpenMC and batch script input files
    generate_input_files(prob_type, seed, cluster, run_file)
