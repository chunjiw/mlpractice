import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", app_title="MovieLens")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## MovieLens Dataset""")
    return


@app.cell
def _():
    from pathlib import Path
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    pd.options.mode.copy_on_write = True
    return Path, defaultdict, np, pd


@app.cell
def _(Path):
    dpath = Path("../data/movielens/ml-latest-small/")
    list(dpath.iterdir())
    return (dpath,)


@app.cell
def _(dpath, pd):
    ratings = pd.read_csv(dpath / 'ratings.csv')
    return (ratings,)


@app.cell
def _(ratings):
    ratings.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Encode data""")
    return


@app.cell
def _(np, ratings):
    np.random.seed(3)
    mask = np.random.rand(len(ratings)) < 0.8
    train = ratings[mask]
    val = ratings[~mask]
    return train, val


@app.cell
def _(ratings):
    ratings.head()
    return


@app.cell
def _(train):
    train.head()
    return


@app.cell
def _(val):
    val.head()
    return


@app.cell
def _(defaultdict, pd):
    a = pd.Series([1,2])
    mapping = defaultdict(lambda:-1, {1:10})
    a.map(mapping)
    return


@app.cell
def _(defaultdict, pd):
    class IdEncoder:
        """Sklearn-like encoder for userId/movieId integer mapping."""
    
        def __init__(self):
            self.mappings = {}
            self.num_uniques = {}

        def fit(self, df: pd.DataFrame, cols=("userId", "movieId")):
            for col in cols:
                uniq = df[col].unique()
                self.num_uniques[col] = len(uniq)
                self.mappings[col] = defaultdict(lambda:-1, {o: i for i, o in enumerate(uniq)})
            return self

        def transform(self, df: pd.DataFrame, cols=("userId", "movieId")) -> pd.DataFrame:
            for col in cols:
                mapping = self.mappings[col]
                df[col] = df[col].map(mapping)
                df = df[df[col] != -1]
            return df

        def fit_transform(self, df: pd.DataFrame, cols=("userId", "movieId")) -> pd.DataFrame:
            return self.fit(df, cols).transform(df, cols)
    return (IdEncoder,)


@app.cell
def _(IdEncoder, train, val):
    encoder = IdEncoder()
    df_train = encoder.fit_transform(train)
    df_val = encoder.transform(val)
    return df_train, df_val


@app.cell
def _(df_train):
    df_train.head()
    return


@app.cell
def _(df_val):
    df_val.tail()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
