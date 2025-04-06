# Comparison of Selection Strategies in Genetic Algorithms: Roulette, Ranking, and Tournament

## How to reproduce

### Prerequisites

* Git
* Python >= 3.10
* Jupyter
* uv >= 0.5.21 __(optional)__

### Generating Results

1. Clone the repository

```sh
git clone 
```

2. Install dependencies

With `uv` *(recommended)*:
```sh
uv install
```

With `pip`:
```sh
pip install -r requirements.txt
```

3. Run the script `main.py`

With `uv` *(recommended)*:
```sh
uv run src/main.py
```

Without `uv`:
```sh
python src/main.py  # if error, try to replace command "python" by "python3"
```

### Analyzing Results

1. Start Jupyter

With `uv`:
```sh
uv run jupyter notebook
```

Without `uv`:
```sh
jupyter notebook
```

2. Open jupyter on web browser (http://localhost:8888/) and fill token with the given value from terminal output
3. Navigate to `src/analysis.ipynb` notebook and open it
4. Checkout results. Feel free to interact with analysis notebook

<!-- ### Parameters -->
