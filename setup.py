import re
from pathlib import Path
from setuptools import find_packages, setup


ROOT_PATH = Path(__file__).parent

LICENSE_PATH = ROOT_PATH / 'LICENSE'

README_PATH = ROOT_PATH / 'README.rst'

REQUIREMENTS_PATH = ROOT_PATH / 'requirement' / 'main.txt'


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
    name='aequitas',
    version='0.26.0',
    # description="",
    long_description=README_PATH.read_text(),
    author="Center for Data Science and Public Policy",
    author_email='datascifellows@gmail.com',
    url='https://github.com/dssg/aequitas-public',
    packages=find_packages('src', exclude=['tests', 'tests.*']),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    license=LICENSE_PATH.read_text(),
    zip_safe=False,
    keywords='bias aequitas',
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
    },
    # test_suite='tests',
    # tests_require=REQUIREMENTS_TEST,
)
