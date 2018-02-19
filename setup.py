#!/usr/bin/env python

from distutils.core import setup

setup(name='medgpc',
      version='1.0',
      description='MedGP',
      author='Li-Fang Cheng',
      author_email='lifangc@princeton.edu',
      license = "BSD 3-clause",
      packages = ["medgpc",
                  "medgpc.clustering",
                  "medgpc.util",
                  "medgpc.visualization",
                  "medgpc.evaluation",
                  ],
      scripts=['medgpc/clustering/run_kernel_clustering.py',
               'medgpc/evaluation/run_medgpc_eval.py',
               'medgpc/util/run_exp_generator.py',
                  ],
      package_dir={'medgpc': 'medgpc'},
      py_modules = ['medgpc.__init__'],
     )