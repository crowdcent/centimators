"""Benchmark tasks: public text-classification sets with fixed, seeded splits.

Every runner (API models on one machine, local models on another) calls
`load_task` so all methods score the exact same test rows.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
CACHE = Path(__file__).parent / ".cache"
SEED = 0
N_TEST = 500
N_TRAIN_POOL = 4000


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    repo: str
    train_file: str
    test_file: str
    text_col: str
    label_col: str
    labels: dict[int, str]  # raw label id -> class name
    descriptions: dict[str, str]  # class name -> one-line meaning
    instructions: str
    question: str

    @property
    def classes(self) -> list[str]:
        return list(self.labels.values())


TASKS = {
    t.name: t
    for t in [
        Task(
            name="onion",
            title="Satire detection (The Onion vs HuffPost)",
            repo="raquiba/Sarcasm_News_Headline",
            train_file="train.json",
            test_file="test.json",
            text_col="headline",
            label_col="is_sarcastic",
            labels={0: "real", 1: "satire"},
            descriptions={
                "real": "a genuine news headline",
                "satire": "a satirical headline from The Onion",
            },
            instructions="Decide whether a news headline is satire.",
            question="Is this headline real news or satire from The Onion?",
        ),
        Task(
            name="sst2",
            title="Movie review sentiment (SST-2)",
            repo="stanfordnlp/sst2",
            train_file="data/train-00000-of-00001.parquet",
            test_file="data/validation-00000-of-00001.parquet",
            text_col="sentence",
            label_col="label",
            labels={0: "negative", 1: "positive"},
            descriptions={
                "negative": "the reviewer dislikes the movie",
                "positive": "the reviewer likes the movie",
            },
            instructions="Classify the sentiment of a movie review snippet.",
            question="Is this movie review snippet negative or positive?",
        ),
        Task(
            name="agnews",
            title="News topic (AG News, 4 classes)",
            repo="fancyzhx/ag_news",
            train_file="data/train-00000-of-00001.parquet",
            test_file="data/test-00000-of-00001.parquet",
            text_col="text",
            label_col="label",
            labels={0: "world", 1: "sports", 2: "business", 3: "scitech"},
            descriptions={
                "world": "world news, politics, conflicts, international affairs",
                "sports": "sports",
                "business": "business, markets, companies, economy",
                "scitech": "science and technology",
            },
            instructions="Classify the topic of a news article.",
            question="Which section does this news article belong to?",
        ),
        Task(
            name="fintweets",
            title="Financial tweet sentiment (3 classes)",
            repo="zeroshot/twitter-financial-news-sentiment",
            train_file="sent_train.csv",
            test_file="sent_valid.csv",
            text_col="text",
            label_col="label",
            labels={0: "bearish", 1: "bullish", 2: "neutral"},
            descriptions={
                "bearish": "negative for the stock, company or market",
                "bullish": "positive for the stock, company or market",
                "neutral": "no clear positive or negative market view",
            },
            instructions="Classify the market sentiment of a financial news tweet.",
            question="Is this financial tweet bearish, bullish, or neutral?",
        ),
    ]
}


def _read(task: Task, file: str) -> pl.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{task.name}-{Path(file).name}"
    if not path.exists():
        with urllib.request.urlopen(HF.format(repo=task.repo, file=file)) as resp:
            path.write_bytes(resp.read())
    readers = {
        ".json": pl.read_ndjson,
        ".csv": pl.read_csv,
        ".parquet": pl.read_parquet,
    }
    df = readers[path.suffix](path)
    return df.select(
        pl.col(task.text_col).alias("text"),
        pl.col(task.label_col).replace_strict(task.labels).alias("label"),
    ).unique("text", keep="first", maintain_order=True)


def load_task(name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(train_pool, test) with columns `text`, `label` (class names)."""
    task = TASKS[name]
    test = _read(task, task.test_file)
    test = test.sample(min(N_TEST, len(test)), seed=SEED, shuffle=True)
    train = _read(task, task.train_file).filter(~pl.col("text").is_in(test["text"]))
    train = train.sample(min(N_TRAIN_POOL, len(train)), seed=SEED, shuffle=True)
    return train, test
