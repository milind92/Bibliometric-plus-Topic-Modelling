"""Example script for performing topic modeling and building a knowledge graph
from PDF files of journal articles in Google Colab.

The workflow is:
1. Install required libraries.
2. Upload PDF files using the Colab file upload widget.
3. Extract text from each PDF.
4. Preprocess the text and run LDA topic modelling with gensim.
5. Extract named entities with spaCy and build a co-occurrence knowledge graph
   using networkx.
6. Visualize the graph interactively with pyvis.

Run each section in separate Colab cells if desired.
"""

# Install required packages (run once)
# In Colab it is common to install in the notebook; here we show the command
# for completeness. Uncomment if running locally.
# !pip install PyPDF2 nltk gensim spacy networkx pyvis

import os
from typing import List

from google.colab import files  # type: ignore
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import spacy
import networkx as nx
from pyvis.network import Network


# Ensure required NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")

# Load spaCy English model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


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
    """Tokenize text and remove stop words and non alphabetic tokens."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.isalpha() and t not in stop_words]


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

# 4. Build a simple knowledge graph from named entity co-occurrence
G = nx.Graph()
for text in texts:
    doc = nlp(text)
    for sent in doc.sents:
        entities = [e.text for e in sent.ents if e.label_ in {"PERSON", "ORG", "GPE"}]
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
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
