# FRDC-ML

**Forest Recover Digital Companion** Machine Learning Pipeline Repository

This repository contains all code regarding our models used.
This is part of the entire E2E pipeline for our product.

_Data Collection -> **FRDC-ML** -> [FRDC-UI](https://github.com/Forest-Recovery-Digital-Companion/FRDC-UI)_

Currently, it's heavily WIP.

# Dev Info

```
FRDC-ML/
    src/                    # All relevant code
        frdc/               # Package/Component Level code
            load/           # Image I/O
            preprocess/     # Image Preprocessing
            train/          # ML Training
            evaluate/       # Model Evaluation
            ...             # ...
        main.py             # Pipeline Entry Point

    tests/                  # PyTest Tests
        integration-tests/  # Tests that run the entire pipeline
        unit-tests/         # Tests for each component

    poetry.lock             # Poetry managed environment file
    pyproject.toml          # Project-level information: requirements, settings, name, deployment info

    .github/                # GitHub Actions
```

## Our Architecture

This is a classic, simple Python Package architecture, however, we **HEAVILY EMPHASIZE** encapsulation of each stage.
That means, there should never be data that **IMPLICITLY** persists across stages. We enforce this by our
`src/main.py` entrypoint.

Each function should have a high-level, preferably intuitively english naming convention.

```python
from torch.optim import Adam

from frdc.load.dataset import FRDCDataset
from frdc.preprocess.morphology import remove_small_objects
from frdc.preprocess.morphology import watershed
from frdc.train import train

ar = FRDCDataset("chestnut", "date", ...)
ar = watershed(ar)
ar = remove_small_objects(ar, min_size=100)
model = train(ar, lr=0.01, optimizer=Adam, )
...
```

This architecture allows for

1) Easily legible high level pipelines
2) Flexibility
    1) Conventional Python signatures can be used to input arguments
    2) If necessary we can leverage everything else Python
3) Easily replicable pipelines

> Initially, we evaluated a few ML E2E solutions, despite them offering great functionality, their flexibility was
> limited. From a dev perspective, **Active Learning** was a gray area, and we foresee heavy shoehorning.
> Ultimately, we decided that the risk was too great, thus we resort to creating our own solution.

## Contributing

### Pre-commit Hooks

We use Black and Flake8 as our pre-commit hooks. To install them, run the following commands:

```bash
poetry install
pre-commit install
```

If you're using `pip` instead of `poetry`, run the following commands:

```bash
pip install pre-commit
pre-commit install
```
