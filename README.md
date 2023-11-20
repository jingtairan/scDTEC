# Unsupervised Deep Topology Embedded Characterization of Single-Cell Chromatin Accessibility Profiles

`scDTEC` is a python package for classifying cell clusters based on chromatin accessibility from scATAC-seq using a deep clustering model based on graph neural networks.

- [Overview](#overview)
- [System Requirments](#system-requirments)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [Data Availability](#data-availability)
- [License](#license)

# Overview
Cell clustering plays a crucial role in the analysis of single-cell Assay for Transposase-Accessible Chromatin using sequencing (scATAC-seq) data, where chromatin features unveil intercellular variability in gene regulation. Single-cell deep embedding models have gained significant popularity for learning feature representations in a low-dimensional space to facilitate clustering. However, these models are susceptible to technical artifacts, noise, and missing values, which can adversely affect their overall performance. To address those limitations, we propose the \textbf{S}ingle-\textbf{C}ell \textbf{D}eep \textbf{T}opology \textbf{E}mbedded \textbf{C}haracterization (scDTEC) model, which obtains a low-dimensional fused representation of chromatin accessibility profiles and cell topological information. scDTEC employs a topology variational autoencoder to transform high-dimensional input data into latent layer representations that combine chromatin accessibility profiles with cellular topological information and then generate reconstructed chromatin information on the decoder. Concurrently, scDTEC employs a contrastive loss to maximize the consistency between the anchor graph derived from the row data and the learning graph generated by the graph learner model and uses the anchor graph as the learning objective. Finally, scDTEC employs a joint optimization paradigm to simultaneously optimize the embedding of cell fusion information and the updating of cell topology structure, guiding the precise partitioning of cell clusters. The results of our evaluation demonstrate the superiority of scDTEC over a variety of cutting-edge methods.
# System Requirments

# Installation Guide

# Usage

# Data Availability
All supporting source code and data can be downloaded from <a href="https://github.com/jingtairan/scDTEC">here</a> and <a href="">Zenodo</a>, and <a href="">FigShare</a>.

# License
This project is covered under the **MIT License**.


Thank you for using scDTEC! Any questions, suggestions or advice are welcome!
Contact:  yichang@jlu.edu.cn, lixt314@jlu.edu.cn, jingtr21@mails.jlu.edu.cn
