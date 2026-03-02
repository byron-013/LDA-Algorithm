# LDA-Algorithm

A from-scratch implementation of **Latent Dirichlet Allocation (LDA)** with collapsed Gibbs sampling for unsupervised topic discovery in text corpora.

## How LDA Works

LDA is a generative probabilistic model that assumes each document is a mixture of latent topics, and each topic is a distribution over words. Given a corpus, the algorithm infers these hidden topic structures by iteratively reassigning words to topics via Gibbs sampling, converging on distributions that best explain the observed word co-occurrence patterns.

## Features

- **Custom Gibbs sampler** — full LDA inference implemented in NumPy, no black-box calls
- **PDF and CSV ingestion** — build corpora from mixed file types using `pdfplumber`
- **Hyperparameter tuning** — grid search over *K*, *α*, and *β* using perplexity as the scoring metric
- **Topic visualization** — horizontal bar charts of top words per topic and a document–topic heatmap
- **Built-in demo corpus** — 24 synthetic documents across four themes (technology, sports, finance, health) so the notebook runs out of the box

## Setup

```bash
# Clone the repository
git clone https://github.com/byron-013/LDA-Algorithm.git
cd LDA-Algorithm

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Open the Jupyter notebook and run all cells:

```bash
jupyter notebook LDA/LDA.ipynb
```

The notebook ships with a demo corpus so no external data is needed. Switch datasets by changing the `DATASET` variable in the notebook:

```python
# Options: "demo", "ag-news", "20-newsgroups"
DATASET = "demo"
```

To add your own dataset, create a folder under `LDA/datasets/` with a CSV and a `config.json` (see `datasets/ag-news/config.json` for the format). You can also load files directly with `corpus_creator()`:

```python
corpus = corpus_creator(['data/articles.csv', 'data/paper.pdf'])
```

### Stock Datasets

| Dataset | Description | Sampled Docs | Categories |
|---------|-------------|-------------|------------|
| `demo` | Built-in synthetic corpus | 24 | 4 |
| `ag-news` | News articles (World, Sports, Business, Sci/Tech) | 200 | 4 |
| `20-newsgroups` | Classic NLP newsgroup posts (computers, science, politics, etc.) | 200 | 20 |

## Example Output

After running the full pipeline you will see:

1. **Grid search log** — perplexity scores for each hyperparameter combination
2. **Top words per topic** — bar charts showing the most probable words in each discovered topic
3. **Document–topic heatmap** — a color-coded matrix showing how each document distributes across topics
4. **Summary table** — dominant topic assignment and weight for every document

## Performance & Scalability

This implementation is **educational, not production-grade**. The goal is to expose the internals of LDA and Gibbs sampling — every step is written in plain Python/NumPy so you can read, modify, and learn from it.

The pure-Python Gibbs sampler uses triple-nested loops (iterations x documents x vocabulary), which means runtime scales steeply with corpus size. Approximate runtimes on a typical machine:

| Documents | Vocabulary | Grid Search + Final Model |
|-----------|-----------|--------------------------|
| 24 | ~60 | ~1 minute |
| 200 | ~1,000 | ~3 minutes |
| 200 | ~2,500 | ~12 minutes |
| 500 | ~5,000 | ~90 minutes |

**Recommended limits:** Keep corpora under ~300 documents for interactive use. For larger datasets, the stock configs sample a subset automatically.

For production workloads with thousands or millions of documents, use an optimized library like [`gensim.models.LdaModel`](https://radimrehurek.com/gensim/models/ldamodel.html) or [`scikit-learn`'s `LatentDirichletAllocation`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) — these use compiled C/Cython code and variational inference, making them orders of magnitude faster.

## Dependencies

- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- pdfplumber
- ipykernel

## About the Author

Byron Delaney Jr — Berkeley Applied Mathematics

Contact: byron13@berkeley.edu
LinkedIn: [linkedin.com/in/byron13](https://www.linkedin.com/in/byron13)
