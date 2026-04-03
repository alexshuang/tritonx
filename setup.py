# SPDX-License-Identifier: Apache-2.0


import distutils.command.build
import os
import subprocess

import setuptools.command.build_py
import setuptools.command.develop
from setuptools import setup, find_packages

TOP_DIR = os.path.realpath(os.path.dirname(__file__))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        setuptools.command.build_py.build_py.run(self)


class build(distutils.command.build.build):
    def run(self):
        self.run_command('build_py')


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('build')
        setuptools.command.develop.develop.run(self)


cmdclass = {
    'build_py': build_py,
    'build': build,
    'develop': develop,
}


def get_git_commit_hash(length=8):
    try:
        cmd = ['git', 'rev-parse', f'--short={length}', 'HEAD']
        return "+git{}".format(subprocess.check_output(cmd).strip().decode('utf-8'))
    except Exception:
        return ""


setup(
    name='tritonx',
    version=f"1.0{get_git_commit_hash()}",
    description='An Triton plugin',
    cmdclass=cmdclass,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
        ],
    },
    license='Apache License v2.0',
    author='Shuang Huang',
    author_email='huangshuang@weituinfo.com',
    url='http://192.168.140.14:9080/huangshuang/tritonx',
    install_requires=[],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10']
)
