import glob
import pickle

from pathlib import Path

from aequitas.fairflow.evaluation import Result


def restructure_results(
    result_path: Path,
    target_path: Path = Path("structured_results"),
) -> None:
    """Restructures results from a given path.

    Transforms the structure from the original
    <dataset_name>/<method_name>/results.pickle to <dataset_name>/<method_name>.pickle.

    Parameters
    ----------
    result_path : Path
        Path to the results.
    target_path : Path, optional
        Path to save the restructured results, by default Path("structured_results").
    """
    result_paths = glob.glob(str(result_path / "*" / "*" / "results.pickle"))
    for result_file in result_paths:
        dataset = result_file.split("/")[-3]
        method = result_file.split("/")[-2]
        if not (target_path / dataset).exists():
            (target_path / dataset).mkdir()
        with open(result_file, "rb") as f:
            results = pickle.load(f)
        with open(target_path / dataset / f"{method}.pickle", "wb") as f:
            pickle.dump(results, f)


def read_results(
    result_path: Path,
    structured: bool = False,
) -> dict[str, dict[str, Result]]:
    """Reads results from a given path.

    Parameters
    ----------
    result_path : Path
        Path to the results.
    structured : bool, optional
        Whether the results are pre-structured (i.e., had the restructure_results) or not, by default False.

    Returns
    -------
    dict[str, dict[str, Result]]
        Dictionary of results, where the first key is the dataset and the second key is
        the method.
    """
    results = {}
    if structured:
        result_paths = glob.glob(str(result_path / "*" / "*.pickle"))
    else:
        result_paths = glob.glob(str(result_path / "*" / "*" / "results.pickle"))
    print(result_paths)
    for result_file in result_paths:
        if structured:
            dataset = result_file.split("/")[-2]
            method = result_file.split("/")[-1].split(".")[0]
        else:
            dataset = result_file.split("/")[-3]
            method = result_file.split("/")[-2]
        if dataset not in results:
            results[dataset] = {}
        with open(result_file, "rb") as f:
            results[dataset][method] = pickle.load(f)
    return results