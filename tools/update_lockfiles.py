import argparse
import os
import subprocess
import sys


try:
    import conda_lock
except:
    print("conda-lock must be installed.")
    exit(1)

parser = argparse.ArgumentParser(
    "Iris Lockfile Generator",
)

parser.add_argument('files', nargs='+', 
    help="List of environment.yml files to lock")
parser.add_argument('--output-dir', '-o', default='.', 
    help="Directory to save output lock files")

args = parser.parse_args()

for infile in args.files:
    print(f"generating lockfile for {infile}", file=sys.stderr)
    fname = os.path.basename(infile)
    ftype = fname.split('.')[-1]
    if ftype.lower() in ('yaml', 'yml'):
        fname = '.'.join(fname.split('.')[:-1])
    
    ofile_template = os.path.join(args.output_dir, fname+'-{platform}.lock')
    subprocess.call([
        'conda-lock',
        'lock',
        '--filename-template', ofile_template,
        '--file', infile,
        '--platform', 'linux-64'
    ])
    print(f"lockfile saved to {ofile_template}".format(platform='linux-64'),
        file=sys.stderr)