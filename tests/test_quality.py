import os
import subprocess


def test_code_linting():
    """Vérifie que le code respecte les standards PEP8 via Ruff"""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "."], capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    ), f"Erreurs de linting détectées :\n{result.stdout}\n{result.stderr}"


def test_code_formatting():
    """Vérifie que le code est correctement formaté via Black"""
    result = subprocess.run(
        ["uv", "run", "python", "-m", "black", "--check", "."],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert (
        result.returncode == 0
    ), f"Le code n'est pas formaté avec Black. Lancez 'black .'\n{result.stdout}\n{result.stderr}"
