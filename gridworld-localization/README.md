# COMS W4701 AI HW4 Programming


### Setup
1. Create and activate your virtual environment (make sure your python is at version 3.9 or higher).
```
# using venv
python -m venv .venv
source .venv/bin/activate   

# on windows
python -m venv .venv
.venv\scripts\activate

# alternatively, using conda
conda create -n hw4env python=3.9 pip
conda activate hw4env
```

2. Install poetry
```
pip install poetry
```

3. Install the `hw4` package
```
poetry install
```

###  Running your implementation
You may run the main program with
```
python main.py
```
