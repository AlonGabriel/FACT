import argparse
import os

import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resumes experiments from a given project.')
    parser.add_argument('project', type=str, help='Project name')
    parser.add_argument('--state', type=str, nargs='+', required=True)
    parser.add_argument('--exclude', type=str, nargs='+')
    parser.add_argument('--job-name', type=str, required=True)
    parser.add_argument('--time-limit', type=str, default='2:00:00')
    parser.add_argument('--directives', type=str, nargs='+')
    parser.add_argument('--working-dir', type=str, default=os.getcwd())
    args = parser.parse_args()

    excluded_tags = args.exclude or []

    directives = '\n'.join(f'#SBATCH --{el}' for el in (args.directives or []))

    with open('slurm/job.template.sh') as f:
        template = f.read()

    iterator = wandb.Api().runs(args.project, per_page=1)
    while iterator.more:
        run = iterator.next()
        if run.state in args.state:
            if any(tag in run.tags for tag in excluded_tags):
                continue
            job = template.format(
                job_name=args.job_name,
                time=args.time_limit,
                directives=directives,
                config=' '.join(f'{key}={value}' for key, value in run.config.items()),
            )
            filepath = f'slurm/jobs/{run.name}.sh'
            with open(filepath, 'w') as f:
                f.write(job)
            command = f'sbatch {filepath} --workdir={args.working_dir}'
            print('\033[93m', 'Running', command, '\033[0m')
            os.system(command)
