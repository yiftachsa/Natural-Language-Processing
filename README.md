# Natural-Language-Processing

<p align="center">
<img src="https://www.cxtoday.com/wp-content/uploads/2021/06/Natural-Language-Processing-1.png"  width="580" height="250">
</p>

This repository contains the tasks created as part of the Natural Language Processing (NLP) course at the Ben-Gurion University of the Negev

Natural language processing is the research field in which we develop, test and analyze machine learning algorithms that are used in order to **automatically process large amounts of text in order to understand given texts and generate new texts.** The course makes heavy use of machine learning but introduces concepts from linguistics and cognitive psychology. Typical examples for active research topics and applications are **spam detection, error correction, machine translation, topic modeling, document classification and demographic attribution.**

---
## Assignments
<p float="left">
  <img src="Media/ass3_1.png" width=30% />
  <img src="Media/ass3_9.png" width=24% />
  <img src="Media/ass3_3.png" width=27% /> 
  <img src="Media/ass3_2.png" width=15% />
</p>

[Assignment 1](Assignments\Assignment1\ex1.py) - Text Preprocessing, Language Modeling and Generation - Implement a Markovian language model and a language generator. We use noisy channel algorithm for spell checking. Combining the noisy channel with a language model is a simple, though powerful, algorithm that demonstrates some key elements in language processing and the way statistical machine learning implicitly accounts for cognitive and technological biases. 

[Assignment 2](Assignments\Assignment2\ex2.py) - Contextual Spell Checking - The Noisy Channel and a Probabilistic Spell Checker. Distributional semantics and Text Classification.  In this assignment we built a spell checker that handles both non-word and real-word errors given in a sentential context. In order to do that we learn a language model as well as reconstruct the error distribution tables (according to error type) from lists of common errors. Finally, we combine it all to a context sensitive noisy channel model.

[Assignment 3](Assignments\Assignment3\ex3.py) | [Notebook](Assignments\Assignment3\NLP_ass3.ipynb) | [Report](Assignments\Assignment3\report.pdf)- Authorship Attribution - LSTM networks - Using various algorithms for text classification, performing an **authorship attribution task on Donald Trump’s tweets.** A comprehensive report, the accompanying code and classification output obtained on a test set is included in the repository. 

[Assignment 4](Assignments\Assignment4\tagger.py) | [Notebook](Assignments\Assignment4\NLP_ass4.ipynb) - Part of Speech Tagging - Implement a Hidden Markov Model and a BiLSTM model for Part of Speech tagging. Using discriminative models for POS tagging (MEMM and bi-LSTM). 


<p align="center">
<img src="https://in.bgu.ac.il/marketing/DocLib/Pages/graphics/heb-en-arabic-logo-small.png">
</p>
