# Long Document Multi-Task Binary Classification Model

## Overview

This repository implements a long-document, multi-task binary classification system built on top of a transformer encoder with parameter-efficient fine-tuning and learned chunk aggregation.

The system is designed to handle documents that exceed transformer token limits by splitting them into overlapping chunks, encoding each chunk independently, and aggregating chunk representations into a single document-level representation using attention pooling. Each document is then classified using a task-specific binary head.

### The architecture prioritizes:

- Scalability to long documents

- Task isolation in multi-task settings

- Training efficiency

- Auditability and governance readiness

## Problem Statement

Transformer models such as BERT and DistilBERT are limited to fixed maximum input lengths (typically 512 tokens). In enterprise settings, documents often exceed this limit, leading to information loss when naïve truncation is applied.

### Additional challenges include:

- Preserving context across chunk boundaries

- Avoiding task interference in multi-task learning

- Controlling training cost and memory usage

- Ensuring explainability and governance compliance

## Solution Summary

This implementation addresses these challenges through:

- Sliding-window chunking with overlap (stride)

- Independent chunk encoding using a LoRA-adapted transformer

- Shared representation learning across tasks

- Attention-based aggregation of chunk embeddings

- Task-specific binary classification heads

## High-Level Architecture

Raw Document Text  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
Tokenizer + Sliding Window Chunking  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
Transformer Encoder (DistilBERT + LoRA)  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
CLS Embedding per Chunk  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
Shared Projection Layer  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
Attention Pooling (per document)  
&emsp;&emsp;&emsp;&emsp;│  
&emsp;&emsp;&emsp;&emsp;▼  
Task-Specific Binary Classification Head
