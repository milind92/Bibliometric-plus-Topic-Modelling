"""Example script for performing topic modelling and building a knowledge graph
from PDF files of journal articles in Google Colab.

It is inspired by the more comprehensive ``V3. Topic Modelling and Knowledge
Graph.txt`` notebook and focuses on an easy to run Colab version.

The workflow is:
1. Install required libraries.
2. Upload PDF files using the Colab file upload widget.
3. Extract text from each PDF.
4. Preprocess the text and run LDA topic modelling with gensim.
5. Extract named entities using a Hugging Face model and build a co-occurrence
   knowledge graph with networkx.
6. Visualize the graph interactively with pyvis.

Run each section in separate Colab cells if desired.
"""

# Install required packages (run once)
# In Colab it is common to install in the notebook; here we show the command
# for completeness. Uncomment if running locally.
# !pip install PyPDF2 nltk gensim networkx pyvis transformers sentencepiece

import os
from typing import List

from google.colab import files  # type: ignore
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import networkx as nx
from pyvis.network import Network
from transformers import pipeline
from standardize_terms import standardize_text


# Ensure required NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load spaCy English model

# Hugging Face NER pipeline for knowledge graph extraction
ner_pipeline = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

# Lemmatizer setup
lemmatizer = WordNetLemmatizer()


def upload_pdfs() -> List[str]:
    """Use the Colab uploader to get PDF files and return a list of filenames."""
    uploaded = files.upload()
    return list(uploaded.keys())


def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + " "
    return text


def preprocess(text: str) -> List[str]:
    """Apply normalization, tokenize, lemmatize and remove stop words."""
    text = standardize_text(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    processed = []
    for tok in tokens:
        if tok.isalpha() and tok not in stop_words:
            processed.append(lemmatizer.lemmatize(tok))
    return processed


# --- Main execution ---

# 1. Upload PDFs
pdf_files = upload_pdfs()

# 2. Extract and preprocess texts
texts = []
processed_docs = []
for pdf in pdf_files:
    raw = extract_text_from_pdf(pdf)
    texts.append(raw)
    processed_docs.append(preprocess(raw))

# 3. Topic modelling using gensim LDA
if processed_docs:
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    for idx, topic in lda_model.print_topics():
        print(f"Topic {idx}: {topic}")
else:
    print("No documents to model.")

# 4. Build a simple knowledge graph using Hugging Face NER results
G = nx.Graph()
for text in texts:
    for sent in sent_tokenize(text):
        ents = [ent["word"] for ent in ner_pipeline(sent)
                if ent["entity_group"] in {"PER", "ORG", "LOC"}]
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                a, b = ents[i], ents[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

# 5. Visualize the knowledge graph with pyvis
net = Network(notebook=True)
for node in G.nodes():
    net.add_node(node, label=node)
for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data["weight"])

net.show("knowledge_graph.html")
print("Knowledge graph saved to knowledge_graph.html")
