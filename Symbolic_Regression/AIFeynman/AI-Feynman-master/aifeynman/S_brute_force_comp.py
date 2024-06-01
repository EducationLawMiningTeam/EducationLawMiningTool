# runs BF on data and saves the best RPN expressions in results.dat. This is used to find function proportional to the desired one using gradients
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


def brute_force_comp(pathdir, filename, BF_try_time, BF_ops_file_type, sigma=10, band=0,
                     og_pathdir=""):

    try_time = BF_try_time
    try_time_prefactor = BF_try_time
    file_type = BF_ops_file_type

    try:
        os.remove(og_pathdir+"results_comp.dat")
    except:
        pass

    try:
        os.remove(og_pathdir+"brute_solutions.dat")
    except:
        pass

    try:
        os.remove(og_pathdir+"brute_constant.dat")
    except:
        pass

    try:
        os.remove(og_pathdir+"brute_formulas.dat")
    except:
        pass

    print("Trying to solve mysteries with brute force...")
    print("Trying to solve {}".format(pathdir+filename))

    shutil.copy2(pathdir+filename, og_pathdir+"mystery.dat")

    data = "'{FT}' '{R}' mystery.dat results_comp.dat {SIG} {BND}".format(
            FT=_get_resource(file_type),
            R=_get_resource("arity2templates.txt"),
            SIG=sigma,
            BND=band
            )

    arg_file = og_pathdir+'args.dat'
    print('writing',arg_file)
    with open(arg_file, 'w') as f:
        f.write(data)

    oldcwd = os.getcwd()
    os.chdir(og_pathdir)
    try:
        subprocess.call(["feynman_sr_mdl4"], timeout=try_time)
    except Exception as e:
        print('Error:',e)
        pass
    os.chdir(oldcwd)

    return 1
