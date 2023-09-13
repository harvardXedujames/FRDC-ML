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
from frdc.load import load_image
from frdc.preprocess import watershed, remove_small_blobs
...
from frdc.train import train
from torch.optim import Adam

ar = load_image("my_img.png")
ar = watershed(ar)
ar = remove_small_blobs(ar, min_size=50)
...
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
