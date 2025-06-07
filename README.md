# Bibliometric-plus-Topic-Modelling

This repository contains experiments on bibliometric analysis, topic modelling
and knowledge graph generation. The notebooks and scripts illustrate different
approaches for processing research papers.

## New: `colab_pdf_topic_kg.py`

`colab_pdf_topic_kg.py` is a minimal example designed for Google Colab. It lets
you upload PDF articles, normalises terminology, performs LDA topic modelling
and builds an interactive knowledge graph. Named entities are detected using a
Hugging Face model and visualised with pyvis.

Run it in Colab cell by cell or as a standalone script after installing the
required packages.

The repository also provides `standardize_terms.py` with an extensive map of
alternate spellings and terminology to ensure consistent text processing.
