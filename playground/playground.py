import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")

with app.setup:
    from centimators.data_transformers import RankTransformer


@app.cell
def _():
    ranker = RankTransformer()
    ranker
    return


if __name__ == "__main__":
    app.run()
