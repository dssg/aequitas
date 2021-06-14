import re
from pathlib import Path
from setuptools import find_packages, setup
import sys
import shutil

ROOT_PATH = Path(__file__).parent

LICENSE_PATH = ROOT_PATH / 'LICENSE'

README_PATH = ROOT_PATH / 'README.md'

REQUIREMENTS_PATH = ROOT_PATH / 'requirement' / 'main.txt'

MANIFEST_PATH = ROOT_PATH / 'MANIFEST.in'

BUILD_PATH = ROOT_PATH / 'build'

NAME = 'aequitas'

EXCLUDE_LIST = ['tests', 'tests.*']

if '-l' in sys.argv or '--lite' in sys.argv:
    NAME += '-lite'
    EXCLUDE_LIST += [
        'aequitas_cli',
        'aequitas_cli.*',
        'aequitas_webapp',
        'aequitas_webapp.*',
    ]
    SELECTED_REQUIREMENTS_PATH = ROOT_PATH / 'requirement' / 'lite.txt'
    SELECTED_MANIFEST_PATH = ROOT_PATH / 'LITE_MANIFEST.in'
    try:
        sys.argv.remove('-l')
    except ValueError:
        sys.argv.remove('--lite')

else:
    SELECTED_REQUIREMENTS_PATH = ROOT_PATH / 'requirement' / 'full.txt'
    SELECTED_MANIFEST_PATH = ROOT_PATH / 'FULL_MANIFEST.in'

try:
    shutil.rmtree(BUILD_PATH)
except OSError as e:
    print(e)

shutil.copy(SELECTED_MANIFEST_PATH, MANIFEST_PATH)
shutil.copy(SELECTED_REQUIREMENTS_PATH, REQUIREMENTS_PATH)

# with open(README_PATH, encoding='utf-8') as f:
#    long_description = f.read()

long_description = """
Aequitas is an open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive tools."""

def stream_requirements(fd):
    """For a given requirements file descriptor, generate lines of
    distribution requirements, ignoring comments and chained requirement
    files.

    """
    for line in fd:
        cleaned = re.sub(r'#.*$', '', line).strip()
        if cleaned and not cleaned.startswith('-r'):
            yield cleaned


with REQUIREMENTS_PATH.open() as requirements_file:
    REQUIREMENTS = list(stream_requirements(requirements_file))


setup(
    name=NAME,
    version='0.42.0',
    description="The bias and fairness audit toolkit.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/aequitas',
    packages=find_packages('src', exclude=EXCLUDE_LIST),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    license='https://github.com/dssg/aequitas/blob/master/LICENSE',
    zip_safe=False,
    keywords='fairness bias aequitas',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'aequitas-report=aequitas_cli.aequitas_audit:main',
        ],
    }
)
