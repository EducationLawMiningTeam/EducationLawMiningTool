# runs BF on data and saves the best RPN expressions in results.dat
# all the .dat files are created after I run this script
# the .scr are needed to run the fortran code

import csv
import os
import shutil
import subprocess
import sys
from subprocess import call

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .resources import _get_resource


def brute_force_number(pathdir, filename, og_pathdir=''):
    try_time = 2
    file_type = "10ops.txt"

    try:
        os.remove(og_pathdir+"results.dat")
        os.remove(og_pathdir+"brute_solutions.dat")
        os.remove(og_pathdir+"brute_formulas.dat")
    except:
        pass

    print("Trying to solve mysteries with brute force...")
    print("Trying to solve {}".format(pathdir+filename))

    shutil.copy2(pathdir+filename, og_pathdir+"mystery.dat")

    data = "'{FT}' '{R}' mystery.dat results.dat".format(
            FT=_get_resource(file_type),
            R=_get_resource("arity2templates.txt"),
            )

    arg_file = og_pathdir+'args.dat'
    print('writing',arg_file)
    with open(arg_file, 'w') as f:
        f.write(data)

    oldcwd = os.getcwd()
    os.chdir(og_pathdir)
    try:
        subprocess.call(["feynman_sr1"], timeout=try_time)
    except:
        pass
    os.chdir(oldcwd)

    return 1
