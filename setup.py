import re
from pathlib import Path
from setuptools import find_packages, setup


ROOT_PATH = Path(__file__).parent

LICENSE_PATH = ROOT_PATH / "LICENSE"

README_PATH = ROOT_PATH / "README.md"

REQUIREMENTS = {
    f"{req}": ROOT_PATH / "requirements" / f"{req}.txt"
    for req in ["main", "webapp", "cli"]
}

with open(ROOT_PATH / "src/aequitas/version.py", "r") as version_file:
    version = re.search(r'__version__ = "(.*?)"', version_file.read()).group(1)

long_description = README_PATH.read_text()


def stream_requirements(fd):
    """For a given requirements file descriptor, generate lines of
    distribution requirements, ignoring comments and chained requirement
    files.

    """
    for line in fd:
        cleaned = re.sub(r"#.*$", "", line).strip()
        if cleaned and not cleaned.startswith("-r"):
            yield cleaned


for req, path in REQUIREMENTS.items():
    with path.open() as requirements_file:
        REQUIREMENTS[req] = list(stream_requirements(requirements_file))

setup(
    name="aequitas",
    version=version,
    description="The bias and fairness audit toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Center for Data Science and Public Policy",
    author_email="datascifellows@gmail.com",
    url="https://github.com/dssg/aequitas",
    packages=find_packages("src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "": [
            "flow/plots/pareto/template.html",
            "flow/plots/pareto/js/dist/fairAutoML.js",
        ]
    },
    install_requires=REQUIREMENTS["main"],
    extras_require={
        "webapp": REQUIREMENTS["webapp"],
        "cli": REQUIREMENTS["cli"],
    },
    license="MIT",
    zip_safe=False,
    keywords="fairness bias aequitas",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "aequitas-report=aequitas_cli.aequitas_audit:main",
        ],
    },
)
