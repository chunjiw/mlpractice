import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", app_title="MovieLens")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Largely follows https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
    ## MovieLens Dataset
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from pandas import DataFrame
    pd.options.mode.copy_on_write = True

    from pathlib import Path
    from collections import defaultdict
    return DataFrame, Path, defaultdict, np, pd


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
def _(DataFrame, defaultdict):
    class IdEncoder:
        """Sklearn-like encoder for userId/movieId integer mapping."""

        def __init__(self):
            self.mappings = {}
            self.num_uniques = {}

        def fit(self, df: DataFrame, cols=("userId", "movieId")):
            for col in cols:
                uniq = df[col].unique()
                self.num_uniques[col] = len(uniq)
                self.mappings[col] = defaultdict(lambda:-1, {o: i for i, o in enumerate(uniq)})
            return self

        def transform(self, df: DataFrame, cols=("userId", "movieId")) -> DataFrame:
            for col in cols:
                mapping = self.mappings[col]
                df[col] = df[col].map(mapping)
                df = df[df[col] != -1]
            return df

        def fit_transform(self, df: DataFrame, cols=("userId", "movieId")) -> DataFrame:
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Embed layer""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return F, nn, torch


@app.cell
def _(nn):
    embed = nn.Embedding(10, 3)
    return (embed,)


@app.cell
def _(embed, torch):
    b = torch.LongTensor([1,2,0,4,5,1,9])
    embed(b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Matrix factorization model""")
    return


@app.cell
def _(nn):
    class MF(nn.Module):
        def __init__(self, num_users, num_items, emb_size=100):
            super(MF, self).__init__()
            self.user_emb = nn.Embedding(num_users, emb_size)
            self.item_emb = nn.Embedding(num_items, emb_size)
            self.user_emb.weight.data.uniform_(0, 0.05)
            self.item_emb.weight.data.uniform_(0, 0.05)
        
        def forward(self, u, v):
            u = self.user_emb(u)
            v = self.item_emb(v)
            return (u*v).sum(1)
    return (MF,)


@app.cell
def _(df_train):
    df_train['userId'].values
    return


@app.cell
def _(df_train, nn, torch):
    num_users = df_train['userId'].max() + 1
    num_items = df_train['movieId'].max() + 1
    emb_size = 30

    user_emb = nn.Embedding(num_users, emb_size)
    item_emb = nn.Embedding(num_items, emb_size)
    users = torch.LongTensor(df_train['userId'].values.copy())
    items = torch.LongTensor(df_train['movieId'].values.copy())
    return item_emb, items, num_items, num_users, user_emb, users


@app.cell
def _(item_emb, items, user_emb, users):
    U = user_emb(users)
    V = item_emb(items)
    (U*V).sum(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare data epochs""")
    return


@app.cell
def _(df_train, torch):
    from torch.utils.data import TensorDataset, DataLoader

    user_tensor = torch.LongTensor(df_train['userId'].values.copy())
    item_tensor = torch.LongTensor(df_train.movieId.values.copy())
    rating_tensor = torch.FloatTensor(df_train.rating.values.copy())

    batch_size = 1024  # or smaller/larger depending on memory

    train_dataset = TensorDataset(user_tensor, item_tensor, rating_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return (train_loader,)


@app.cell
def _(MF, num_items, num_users):
    model = MF(num_users, num_items)
    return (model,)


@app.cell
def _(F, torch):
    def train_epochs(model, train_loader, epochs=10, lr=0.01, wd=0.0):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        model.train()
    
        for epoch in range(epochs):
            total_loss = 0
            for users, items, ratings in train_loader:
            
                preds = model(users, items)
                loss  = F.mse_loss(preds, ratings)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                total_loss += loss.item() * users.size(0)
        
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    return (train_epochs,)


@app.cell
def _(model, train_epochs, train_loader):
    # Train for 10 epochs, learning rate 0.01, no weight decay
    train_epochs(model, train_loader, epochs=10, lr=0.01, wd=0.0)
    return


if __name__ == "__main__":
    app.run()
