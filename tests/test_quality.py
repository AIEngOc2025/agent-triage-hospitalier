import subprocess


def test_code_linting():
    """Vérifie que le code respecte les standards PEP8 via Ruff"""
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)
    assert result.return_code == 0, f"Erreurs de linting détectées :\n{result.stdout}"

def test_code_formatting():
    """Vérifie que le code est correctement formaté via Black"""
    result = subprocess.run(["black", "--check", "."], capture_output=True, text=True)
    assert result.return_code == 0, "Le code n'est pas formaté avec Black. Lancez 'black .'"
