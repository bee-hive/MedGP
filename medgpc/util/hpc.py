import os
import stat

def write_scheduler_sh(output_dir, output_file_name, scheduler_type, base_cmds,
                       memory='1000', time='2-00:00:00', thread=1, node_thread=20,
                       extra_cmd=None):
    script_full_name = os.path.join(output_dir, output_file_name)
    f = open(script_full_name, 'w')
    f.write('#!/bin/bash' + '\n')
    if(scheduler_type == 'slurm'):
        # medgpc currently support single-node operation
        f.write('#SBATCH -N 1\n')
        f.write('#SBATCH --ntasks-per-node={}\n'.format(thread))
        f.write('#SBATCH -t {}\n'.format(time))
        f.write('#SBATCH --mem={}\n'.format(memory))
    elif(scheduler_type == 'sequential'):
        pass
    elif(scheduler_type == 'pbs'):
        f.write('#PBS -l select=1:ncpus=272 -lplace=excl\n')
        f.write('#PBS -l walltime={}\n'.format(time))
        f.write('#PBS -V\n')
    else:
        raise NotImplementedError

    # f.write('export KMP_ALL_THREADS={}\n'.format(node_thread))
    # f.write('export KMP_DEVICE_THREAD_LIMIT={}\n'.format(node_thread))
    f.write('export OMP_NUM_THREADS={}\n'.format(node_thread))
    f.write('export KMP_AFFINITY=granularity=fine,compact,1,0;\n')
    for ll in base_cmds:
        f.write(ll)
    f.write('\n')
    if(extra_cmd is not None):
        for cc in extra_cmd:
            f.write('{}\n'.format(cc))
    f.close()
    st = os.stat(script_full_name)
    os.chmod(script_full_name, st.st_mode | stat.S_IEXEC)

