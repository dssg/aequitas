# Development Guidelines

Welcome to the development guidelines for the Aequitas project. This document provides an overview of the development process and best practices to follow when working on the project. Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

#### Report Bugs
Report bugs at https://github.com/dssg/aequitas/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### Write Documentation

Aequitas could always use more documentation, whether as part of the official Aequitas docs, in docstrings, or even on the web in blog posts, articles, presentations, and such.

#### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/dssg/aequitas/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome 🙂 

## Version Control
The codebase is officially hosted at [GitHub](https://github.com/dssg/aequitas). Clone or fork the repository and create a new branch for each feature/fix you are working on. Make sure to have an issue associated with the branch you are working on, and reference the issue in any merge requests. When you have completed your work, submit a pull request to the main branch. Provide a clear and detailed description of the changes made and any relevant information for reviewers.

## Setup
For this project, we recommend using [Astral's uv](https://docs.astral.sh/uv/) as the project management tool. To set up the environment for the project, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/dssg/aequitas.git
```
2. Setup the virtual environment:
```bash
cd aequitas
uv run python  # Instantiate a python shell with the venv.
```

Make sure that you have a Python version of 3.8 or higher in your system and this version is detected or managed by uv.

## Formatting
When creating or altering a file, format it using `ruff`. This tool organizes imports (similar to `isort`), formats the code (similar to `black`), and checks for linting errors (similar to `flake8`). To format a file, run:
```bash
uv run ruff format <filename>
``` 
or alternatively, to format all files in the project, run:
```bash
uv run ruff format src
```

## Tests
To run the unit tests included in the package, use the following command:
```bash
uv run --extra cli --extra webapp pytest
```
To run the tests, it is necessary to include the extra dependencies of the package. 
To return a coverage report, run:
```bash
uv run --extra cli --extra webapp pytest --cov-report xml:cov.xml
```
This file can be used by your IDE (e.g. VSCode) to display the coverage of the tests.
Ensure that the code you submit has a good coverage of tests (at least 80%).

## Documentation
TBD.
