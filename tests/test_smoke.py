from pathlib import Path


def test_project_structure_exists():
    # Compute repository root based on this test file location
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "src").is_dir()
    assert (repo_root / "tests").is_dir()
    assert (repo_root / "README.md").is_file()
