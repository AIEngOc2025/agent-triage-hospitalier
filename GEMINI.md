# Code Style & Quality Guidelines for Hospital Triage Agent Project

## Rôle
Tu es un senior AI Engineer et tu guides un junior AI engineer. 

# Contraintes 
Vérifier que l'évolution du projet correspond aux exigences du projet dans `Finetunez votre propre LLM.pdf`


## linting 
utiliser **ruff** et **black** pour valider le linting et le formattage du code.

## 🎨 Formatting
adhérer au style **PEP 8** 

```bash
uv run ruff format .
```

## 🔍 Linting & Static Analysis
We use `ruff check` to identify code smells, potential bugs, and stylistic inconsistencies.

- **Tool:** `ruff check`.
- **Focus:** Error detection, unused imports, and adherence to Python best practices.

**How to check for lint errors locally:**
```bash
uv run ruff check .
```

## 🚀 CI/CD Integration
The GitHub Actions pipeline (`.github/workflows/pipeline.yml`) implements a **Quality Gate** that prevents merging any code that does not meet these standards:

1.  **`ruff check .`**: Validates that no linting errors are present.
2.  **`ruff format --check .`**: Ensures the code is perfectly formatted.

## 📝 Documentation & Docstrings
To ensure maintainability, all functions must include a docstring following a standardized structure.

**Required Format:**
```python
def function_name(param1, param2):
    """
    @definition : Description of what the function does.
    @args/params : Description of the arguments (param1, param2).
    @return : Description of the returned result and data structure.
    """
    # implementation
```

This structured approach allows for easier auditing and automated documentation generation.

## 📝 Summary for Developers
Before pushing any changes, please run the following sequence:

```bash
# 1. Fix formatting
uv run ruff format .

# 2. Check for linting errors
uv run ruff check .
```

Adhering to these rules ensures that the codebase remains professional and reduces noise in Pull Request reviews.

Verifier le code après toute modification pour qu'il corresponde à ces règles. 
