import os
import datetime
from argparse import ArgumentParser

def parse_arguments():
    args = ArgumentParser()

    args.add_argument('--name', type=str)
    args.add_argument('--ngpus', type=int, default=1)
    args.add_argument('--exclude_small_GPUs', action='store_true')
    args.add_argument('--exclude_machines', type=str, default="")
    args.add_argument('--run_local', action='store_true')

    args.add_argument('--root_dir', type=str, default="/cluster/mayr/nbb/vanilla/")
    args.add_argument('--save_dir', type=str, default="/home/mayr/Documents/experiments/EntityTransformer")
    args.add_argument('--books', type=str, default="Band2,Band3,Band4")
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--n_head', type=int, default=1)
    args.add_argument('--line_height', type=int, default=64)
    args.add_argument('--hidden_size', type=int, default=256)
    args.add_argument('--dropout', type=float, default=0.1)
    args.add_argument('--lr', type=float, default=0.0001)
    args.add_argument('--eps', type=float, default=0.)
    args.add_argument('--noise_teacher_forcing', type=float, default=0.2)

    args.add_argument('--augment', action='store_true')
    args.add_argument('--binarize', action='store_true')
    args.add_argument('--wi_features', type=str, default="features_r18_gmp")

    return args.parse_args()



def use_cluster(args):
    mem = 16000 * args.ngpus
    cpus = 4 * args.ngpus
    currentdate = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    command = 'python3 trainEntityTransformerScript.py --name={} --root_dir={} --save_dir={} {} {} --books={} --lr={} --eps={}' \
              ' --noise_teacher_forcing={} --line_height={} --hidden_size={} --batch_size={} --n_head={}' \
        .format(args.name, args.root_dir, args.save_dir, "--augment" if args.augment else "",
                "--binarize" if args.binarize else "", args.books, args.lr, args.eps, args.noise_teacher_forcing,
                args.line_height, args.hidden_size, args.batch_size, args.n_head)
    print(command)

    if args.run_local != True:

        template = ('#!/bin/bash\n'
                    '#SBATCH --job-name={0}\n'
                    '#SBATCH --ntasks=1\n'
                    '#SBATCH --cpus-per-task={1}\n'
                    '#SBATCH --mem={2}\n'
                    '#SBATCH --gres=gpu:{3}\n'
                    '#SBATCH -o /home/%u/slurm-{0}-{4}.out\n'
                    '#SBATCH -e /home/%u/slurm-{0}-{4}.err\n'
                    )
        #template += "pip3 install --user -r requirements_old.txt\n\n"
        template += '#SBATCH --exclude=lme49,lme170 \n' if args.exclude_small_GPUs else ''
        template += '#SBATCH --exclude={} \n'.format(args.exclude_machines) if len(args.exclude_machines) > 0 else ''

        template += "source /cluster/`whoami`/.virtualenvs/nbb/bin/activate \n"
        template += 'echo "Your job is running on" $(hostname)\n'
        template += 'pip3 install -r requirements_cluster.txt \n'

        template += command + '\n'

        submit_path = os.path.join(args.save_dir, args.name)
        os.makedirs(submit_path, exist_ok=True)
        submit_file = os.path.join(submit_path, 'submit.sh')
        with open(submit_file, 'w') as f:
            f.write(template.format(args.name, cpus, mem, args.ngpus, currentdate))

        os.system('sbatch {}'.format(submit_file))

    else:
        os.system(command)



if __name__ == "__main__":
    args = parse_arguments()
    use_cluster(args)