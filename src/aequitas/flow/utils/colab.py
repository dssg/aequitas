# Helper file to help download some objects to Google Colab environment.
import subprocess
from pathlib import Path
from typing import Literal

from aequitas.flow.utils import create_logger


def get_examples(
    directory: Literal[
        "configs",
        "data",
        "experiment_results",
        "methods/data_repair",
    ]
) -> None:
    """Downloads the examples from the fairflow repository.

    Note that this should not be used outside Google Colab, as it clutters the directory
    with with the git files from Aequitas repository.

    Parameters
    ----------
    directory : Literal["configs", "examples/data_repair", "experiment_results"]
        The directory to download from the fairflow repository.
    """
    directory = "examples/" + directory
    logger = create_logger("utils.colab")
    logger.info("Downloading examples from fairflow repository.")
    # Create directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)
    # Check if git repository already exists in folder
    if not Path(".git").exists():
        subprocess.run(
            ["git", "init"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    logger.debug("Git repository initialized.")
    # Check the remote of the git repository
    remote = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"], capture_output=True
    )
    if remote.stdout.decode("utf-8") != "https://github.com/dssg/aequitas\n":
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "-f",
                "origin",
                "https://github.com/dssg/aequitas",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    logger.debug("Git remote checked.")
    # Check if sparse checkout is enabled
    sparse_checkout = subprocess.run(
        ["git", "config", "--get", "core.sparseCheckout"], capture_output=True
    )

    first_run = False
    if sparse_checkout.stdout.decode("utf-8") != "True\n":
        subprocess.run(
            ["git", "config", "core.sparseCheckout", "True"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        Path(".git/info/sparse-checkout").touch()
        first_run = True
    logger.debug("Git sparse checkout checked.")

    # Check if file in sparse checkout
    with open(".git/info/sparse-checkout", "r") as f:
        sparse_checkout_file = f.read()
    if f"{directory}/**" not in sparse_checkout_file:
        sparse_checkout_file += f"\n{directory}/**"
        with open(".git/info/sparse-checkout", "w") as f:
            f.write(sparse_checkout_file)

    # Pull the files
    if not first_run:
        subprocess.run(
            ["git", "read-tree", "-mu", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            "git",
            "pull",
            "origin",
            "master",
        ],  # TODO: Change to main when close to merge
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Examples downloaded.")
