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

The notebook ships with a demo corpus so no external data is needed. To use your own data, replace the demo corpus cell with a call to `corpus_creator()`:

```python
corpus = corpus_creator(['data/articles.csv', 'data/paper.pdf'])
```

## Example Output

After running the full pipeline you will see:

1. **Grid search log** — perplexity scores for each hyperparameter combination
2. **Top words per topic** — bar charts showing the most probable words in each discovered topic
3. **Document–topic heatmap** — a color-coded matrix showing how each document distributes across topics
4. **Summary table** — dominant topic assignment and weight for every document

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
