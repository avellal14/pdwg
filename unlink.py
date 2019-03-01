import argparse
import subprocess
import pdb

def run(cmd):
    subprocess.check_call(cmd, shell=True)

run('rm -rf ./*.py')
run('rm -rf ./*/*/.pyc')
run('rm -rf ./datasetLoaders/*.py')
run('rm -rf ./datasetLoaders/*/*.pyc')
run('rm -rf ./models/*/*.py')
run('rm -rf ./models/*/*/*.pyc')
run('rm -rf ./models_old/*/*.py')
run('rm -rf ./models_old/*/*/*.pyc')
run('rm -rf ./oldcode/*/*.py')
run('rm -rf ./oldcode/*/*/*.pyc')
