# DocuMind AI — Chat with Your PDF Using RAG

An AI-powered PDF question-answering application built with Python, Streamlit, Sentence Transformers, FAISS, and Ollama. DocuMind AI allows users to upload a PDF, ask questions about its content, and review relevant source passages used to generate answers.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Is This a Real AI Project?](#2-is-this-a-real-ai-project)
3. [Features](#3-features)
4. [How It Works](#4-how-it-works)
5. [System Architecture](#5-system-architecture)
6. [Project Structure](#6-project-structure)
7. [Technologies Used](#7-technologies-used)
8. [Core Concepts](#8-core-concepts)
9. [Prerequisites](#9-prerequisites)
10. [Installation and Setup](#10-installation-and-setup)
11. [Run the Application](#11-run-the-application)
12. [How to Use](#12-how-to-use)
13. [Run Tests](#13-run-tests)
14. [Limitations](#14-limitations)
15. [Future Improvements](#15-future-improvements)
16. [What I Learned](#16-what-i-learned)
17. [Interview Explanation](#17-interview-explanation)
18. [Common Issues](#18-common-issues)
19. [Disclaimer](#19-disclaimer)

---

## 1. Project Overview

DocuMind AI is a document question-answering application that uses **Retrieval-Augmented Generation (RAG)** to answer questions using information extracted from PDF documents.

Traditional keyword searches require users to find relevant text themselves. DocuMind AI instead converts document text into numerical representations called embeddings, retrieves relevant passages for a question, and provides those passages to a language model as context.

The application is designed to make information inside documents easier to search and understand.

### Example Use Case

Imagine uploading a 100-page technical document and asking:

- What is the main objective of this document?
- Explain the key concepts discussed in Chapter 2.
- What technologies are mentioned?
- Summarize the important points.
- Which passage supports this answer?

The system retrieves relevant text and uses it as context for generating a response.

**Project type:** AI application / Retrieval-Augmented Generation (RAG)

**Primary language:** Python

**Interface:** Streamlit

**Language model runtime:** Ollama

**Vector search:** FAISS

---

## 2. Is This a Real AI Project?

Yes. DocuMind AI is an AI application that integrates pretrained machine-learning models with information retrieval and language generation.

However, it is important to describe the project accurately.

This project does not train a foundation model from scratch or fine-tune a language model. It uses pretrained models to build a practical AI-powered application.

The project demonstrates experience with:

- Text extraction and preprocessing
- Text chunking and overlap
- Semantic embeddings
- Vector indexing and similarity search
- Retrieval-Augmented Generation
- Prompt construction
- Local language-model integration
- Source passage display
- Application development and testing

These are useful foundations for AI engineering and LLM application development.

---

## 3. Features

- **PDF text extraction:** Extracts readable text from PDF documents.
- **Text chunking:** Splits long documents into smaller, overlapping text segments.
- **Semantic embeddings:** Converts text into numerical vectors using Sentence Transformers.
- **Vector search:** Uses FAISS to retrieve relevant text chunks.
- **Question answering:** Uses retrieved passages as context for a language model.
- **Local LLM integration:** Uses Ollama to run a compatible language model locally.
- **Interactive interface:** Provides a Streamlit interface for document questions and answers.
- **Source inspection:** Displays source passages returned by the retrieval and answer-generation pipeline.
- **Automated tests:** Includes tests for document loading, text splitting, retrieval, vector storage, and answer generation.

Feature availability in the running application depends on the current implementation and configuration.

---

## 4. How It Works

DocuMind AI has two main workflows: document ingestion and question answering.

### A. Document Ingestion

1. The user uploads a PDF.
2. PyMuPDF extracts readable text from the document.
3. The extracted text is divided into smaller chunks.
4. Each chunk is converted into an embedding vector.
5. FAISS stores the vectors in a searchable index.
6. The document chunks and index are used for later retrieval.

### B. Question Answering

1. The user enters a question.
2. The question is converted into an embedding.
3. FAISS searches for relevant document chunks.
4. The retrieved passages are added to the prompt as context.
5. Ollama runs the configured language model to generate an answer.
6. The application displays the answer and the available source passages.

### RAG Workflow Diagram

```text
                  DOCUMENT INGESTION

       Upload PDF
           |
           v
    Extract PDF Text
       (PyMuPDF)
           |
           v
      Split into Chunks
     (Size: 1000 characters,
      Overlap: 150 characters)
           |
           v
   Generate Text Embeddings
    (Sentence Transformers)
           |
           v
      Store in FAISS
           |
           v
      Searchable Index


                  QUESTION ANSWERING

       User Question
           |
           v
   Embed the Question
           |
           v
    Retrieve Relevant
       Text Chunks
        (FAISS)
           |
           v
   Build Contextual Prompt
           |
           v
    Generate an Answer
        (Ollama)
           |
           v
   Display Answer and
      Source Passages
      (Streamlit)
```

The configured chunk size, overlap, and retrieval count are implementation settings. They should be evaluated against representative documents before being treated as optimal.

---

## 5. System Architecture

The application separates document processing, embeddings, vector search, retrieval, and generation into different Python modules.

| Component         | Responsibility                                             |
| ----------------- | ---------------------------------------------------------- |
| Streamlit UI      | Accepts PDFs and questions and displays results            |
| Document loader   | Extracts text from PDF pages                               |
| Text splitter     | Creates overlapping text chunks                            |
| Embedding module  | Converts text into embedding vectors                       |
| Vector store      | Stores and searches vectors using FAISS                    |
| Knowledge base    | Coordinates document ingestion and question answering      |
| RAG pipeline      | Retrieves relevant passages and prepares answer generation |
| Generation module | Communicates with Ollama to generate answers               |
| Test suite        | Tests individual components and expected behavior          |

This modular design makes the application easier to understand, debug, and extend.

---

## 6. Project Structure

The project is organized approximately as follows. The actual files in your local project take precedence if your folder structure differs.

```text
DocuMind AI/
│
├── app.py
├── ask_rag.py
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_stores.py
│   ├── knowledge_base.py
│   ├── rag.py
│   └── generation.py
│
├── tests/
│   ├── test_document_loader.py
│   ├── test_retrieval.py
│   ├── test_text_splitter.py
│   ├── test_vector_store.py
│   └── test_generation.py
│
├── data/
│   └── streamlit_knowledge_base/
│
├── .streamlit/
│   └── config.toml
│
└── .venv/
```

### File Descriptions

**`app.py`**

Contains the Streamlit user interface for uploading documents, asking questions, and viewing answers.

**`ask_rag.py`**

Provides a command-line entry point for interacting with the RAG pipeline, depending on its current implementation.

**`src/document_loader.py`**

Extracts text from PDF files using PyMuPDF and handles document-loading errors.

**`src/text_splitter.py`**

Splits extracted text into manageable chunks with overlap to help preserve context between adjacent chunks.

**`src/embeddings.py`**

Loads and uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert text into embeddings.

**`src/vector_stores.py`**

Contains vector-index functionality based on FAISS and NumPy.

**`src/knowledge_base.py`**

Coordinates PDF ingestion and knowledge-base question answering.

**`src/rag.py`**

Retrieves relevant chunks for a question and passes the retrieved context to the generation component.

**`src/generation.py`**

Communicates with Ollama and generates answers from the supplied context.

**`tests/`**

Contains automated tests for important parts of the application.

**`data/`**

Stores local knowledge-base data or generated artifacts, depending on the implementation.

**`.streamlit/config.toml`**

Contains Streamlit configuration, including the application's theme.

**`.venv/`**

Contains the local Python virtual environment. It should not normally be committed to Git.

---

## 7. Technologies Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Core application logic              |
| Streamlit             | Interactive web interface           |
| PyMuPDF               | PDF text extraction                 |
| Sentence Transformers | Text embedding generation           |
| `all-MiniLM-L6-v2`    | Pretrained embedding model          |
| NumPy                 | Numerical operations                |
| FAISS                 | Efficient vector similarity search  |
| Ollama                | Local language-model runtime        |
| `llama3.2:3b`         | Configured language model           |
| pytest                | Automated testing                   |
| Git and GitHub        | Version control and project sharing |

### Why These Technologies?

**Python:** Provides the core language for document processing, embeddings, retrieval, and AI integration.

**Streamlit:** Allows the application interface to be developed without building a separate frontend framework.

**PyMuPDF:** Extracts text from readable PDF documents.

**Sentence Transformers:** Converts text into vectors that can represent semantic similarities.

**FAISS:** Searches vector representations efficiently.

**Ollama:** Makes it possible to run a supported language model locally.

**pytest:** Helps verify individual components and catch regressions during development.

---

## 8. Core Concepts

### 8.1 Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with language generation.

Instead of asking a language model to answer a question only from its learned parameters, the application retrieves relevant external text and supplies it as context.

RAG has three main stages:

1. **Retrieval:** Find relevant information.
2. **Augmentation:** Add that information to the prompt.
3. **Generation:** Generate a response using the question and context.

RAG can improve answers about specific documents, but it does not guarantee correctness.

### 8.2 Text Chunking

Long documents are divided into smaller sections called chunks.

This makes it easier to retrieve relevant passages without passing an entire document to the model.

The current configuration uses a chunk size of approximately 1000 characters and an overlap of approximately 150 characters.

Overlap allows some text to appear in adjacent chunks, which can help preserve context around chunk boundaries.

These settings are starting points, not universally optimal values.

### 8.3 Embeddings

An embedding is a numerical representation of text.

Semantically related text can have similar vector representations, allowing the system to search for relevant passages even when the question and document do not use exactly the same words.

DocuMind AI uses the pretrained `all-MiniLM-L6-v2` model for embeddings.

The embedding model is pretrained; this project does not train it from scratch.

### 8.4 Vector Search and FAISS

FAISS is a library for efficient similarity search over vectors.

The application embeds a question and searches the vector index to find document chunks that are relevant to that question.

FAISS is a vector-search library, not by itself a complete production database or document-management system.

### 8.5 Language Model

The application uses Ollama to communicate with the configured language model, `llama3.2:3b`.

The model generates an answer using the question and retrieved passages supplied by the application.

The quality of the response depends on the model, retrieved evidence, prompt design, and document quality.

### 8.6 Source Passages

Displaying retrieved passages helps users inspect the evidence behind an answer.

However, showing a passage does not automatically prove that every statement in the generated answer is supported by that passage. Citation and answer-grounding accuracy should be evaluated separately.

---

## 9. Prerequisites

Before running the application, install or prepare:

- Python 3.12 or a compatible Python version
- pip
- Git, if cloning or versioning the project
- Ollama
- The configured language model
- Sufficient disk space for Python packages and model files
- Sufficient RAM and processing resources for the selected models

The first embedding-model download and language-model download can take time.

---

## 10. Installation and Setup

### Step 1: Open the Project Folder

Open PowerShell or the VS Code terminal and navigate to your project directory.

```powershell
cd "$HOME\OneDrive\Desktop\DocuMind AI"
```

If your project is stored elsewhere, replace the path with your actual project location.

### Step 2: Create a Virtual Environment

```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution policy for your environment or activate the environment through the VS Code interpreter selector. Avoid changing system-wide security settings unnecessarily.

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Python Dependencies

Install the packages used by the project:

```powershell
python -m pip install streamlit pymupdf sentence-transformers faiss-cpu numpy pytest ollama
```

If your source code uses a different Ollama integration or additional dependencies, install the packages required by that implementation.

For a reproducible setup, create and maintain a `requirements.txt` file containing the tested dependency versions.

### Step 6: Install Ollama

Install Ollama from its official website:

https://ollama.com/

Follow the installation instructions for your operating system.

### Step 7: Download the Language Model

```powershell
ollama pull llama3.2:3b
```

Check that the model is available:

```powershell
ollama list
```

Make sure the model name matches the one configured in your source code.

### Step 8: Check the Project Files

Confirm that the main application and source modules exist:

```text
app.py
src/document_loader.py
src/text_splitter.py
src/embeddings.py
src/vector_stores.py
src/knowledge_base.py
src/rag.py
src/generation.py
```

---

## 11. Run the Application

From the project root, activate the virtual environment and run:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

Keep the terminal open while using the application.

If the application cannot connect to Ollama, verify that Ollama is installed, running, and has the configured model available.

### Optional: Command-Line Usage

If `ask_rag.py` is implemented for command-line usage in your current version, follow its supported arguments and instructions.

Do not assume that a particular command-line argument is available without checking the file.

---

## 12. How to Use

1. Start the Streamlit application.
2. Open the local URL shown in the terminal.
3. Upload a PDF containing readable text.
4. Wait for the document to be processed and indexed.
5. Enter a question about the document.
6. Submit the question.
7. Review the generated answer.
8. Inspect the retrieved source passages to assess whether they support the response.
9. Try additional questions about different sections of the document.

### Example Questions

For a technical document:

- What is the purpose of this document?
- Explain the main concepts in simple language.
- Which technologies does the document mention?
- Summarize the important points.
- What evidence supports the answer?

For a report:

- What are the main findings?
- Which risks or limitations are described?
- Summarize the conclusions.
- What recommendations are included?

The quality of the answers depends on the extracted text, retrieval quality, prompt design, and language model.

---

## 13. Run Tests

The project includes a pytest test suite for important components.

Run all tests from the project root:

```powershell
python -m pytest -v
```

To run a specific test file:

```powershell
python -m pytest tests/test_document_loader.py -v
```

```powershell
python -m pytest tests/test_text_splitter.py -v
```

```powershell
python -m pytest tests/test_vector_store.py -v
```

```powershell
python -m pytest tests/test_retrieval.py -v
```

```powershell
python -m pytest tests/test_generation.py -v
```

### What Should Be Tested?

- PDF loading and invalid-file handling
- Empty or unreadable PDF content
- Text chunk boundaries and overlap
- Embedding dimensions and vector indexing
- Retrieval of relevant passages
- Generation behavior and error handling
- Streamlit upload and question-answering flow
- Behavior when no useful evidence is retrieved
- Whether results from one PDF can accidentally affect another PDF's answers

Passing unit tests does not, by itself, establish that the complete application works correctly for every document or question.

---

## 14. Limitations

DocuMind AI is a learning and development project. It should not be described as production-ready without further testing and security work.

### 14.1 Scanned PDFs

The PDF extraction pipeline uses PyMuPDF. Scanned documents containing only images may require Optical Character Recognition (OCR).

### 14.2 Hallucinations

The language model can generate incorrect or unsupported information. Retrieved context can help, but it does not eliminate hallucinations.

### 14.3 Retrieval Quality

The application retrieves a configured number of chunks. The best retrieval settings depend on the document, questions, embedding model, and search configuration.

### 14.4 Model Resources

Local inference requires suitable hardware resources. Performance depends on available RAM, CPU, GPU, model size, and configuration.

### 14.5 Evaluation

The project needs a representative evaluation dataset and measured retrieval and answer-quality metrics before claims about accuracy or reliability can be made.

### 14.6 Data Isolation and Persistence

The knowledge-base storage configuration should be tested carefully when processing multiple PDFs. Confirm that old chunks cannot leak into answers for a new document and that stored indexes are managed correctly.

### 14.7 Security and Deployment

Authentication, authorization, upload limits, safe storage, logging, rate limiting, and deployment security require additional work before exposing the application to multiple users.

### 14.8 Current Implementation Status

The README describes the intended architecture and current project configuration. Run the application and test the complete workflow in your environment before claiming every feature has been verified.

---

## 15. Future Improvements

Potential improvements include:

- [ ] Verify the full Streamlit workflow using multiple PDFs.
- [ ] Add a reproducible `requirements.txt` file.
- [ ] Improve error messages and input validation.
- [ ] Add OCR support for scanned PDFs.
- [ ] Evaluate different chunk sizes and overlap settings.
- [ ] Measure retrieval performance on a fixed evaluation dataset.
- [ ] Evaluate answer correctness and evidence support.
- [ ] Improve prompts for insufficient or missing evidence.
- [ ] Add document-specific storage and index management.
- [ ] Add support for multiple documents with clear isolation.
- [ ] Explore hybrid retrieval or reranking if evaluation demonstrates a benefit.
- [ ] Add automated integration tests.
- [ ] Add structured application logging.
- [ ] Improve the interface and source display.
- [ ] Add authentication and upload restrictions if deploying for multiple users.
- [ ] Compare embedding models using measured results.
- [ ] Deploy the application with appropriate security controls.

Prioritize measurable improvements rather than adding complexity without evaluating whether it helps.

---

## 16. What I Learned

This project provides practical experience with:

- Building a Python application with modular components
- Extracting and preprocessing PDF text
- Splitting long text into overlapping chunks
- Using pretrained embedding models
- Understanding vector representations
- Indexing and retrieving vectors with FAISS
- Implementing a Retrieval-Augmented Generation pipeline
- Connecting a local language model through Ollama
- Passing retrieved evidence into a generation prompt
- Building an interactive interface using Streamlit
- Testing Python components with pytest
- Debugging package, import, model, and application issues
- Identifying limitations and planning measurable improvements

These skills form a useful foundation for further work in AI engineering, LLM applications, and information retrieval.

---

## 17. Interview Explanation

### Short Explanation

"DocuMind AI is a PDF question-answering application built using Retrieval-Augmented Generation. It extracts text from PDFs, splits the content into overlapping chunks, generates embeddings using Sentence Transformers, and indexes the vectors with FAISS. When a user asks a question, the application retrieves relevant passages and provides them as context to a locally running language model through Ollama. The Streamlit interface displays the generated answer and available source passages."

### Interview Questions and Answers

**Q1. What problem does your project solve?**

It makes it easier to ask questions about PDF documents without manually searching through every page.

**Q2. What is RAG?**

Retrieval-Augmented Generation combines retrieval of relevant external information with language-model generation. The retrieved information is provided as context when generating an answer.

**Q3. Why did you use embeddings?**

Embeddings represent text numerically, enabling semantic retrieval instead of relying exclusively on exact keyword matches.

**Q4. Why did you use FAISS?**

FAISS provides vector indexing and similarity-search functionality for finding relevant chunks.

**Q5. Did you train your own AI model?**

No. The project uses pretrained embedding and language models. My work focuses on integrating these models into a document question-answering pipeline.

**Q6. Does RAG eliminate hallucinations?**

No. RAG can provide relevant evidence and improve grounding, but the model can still misinterpret context or generate unsupported statements.

**Q7. Why do you split documents into chunks?**

Chunking makes retrieval more focused and avoids needing to supply an entire long document for every question. Overlap helps preserve context near chunk boundaries.

**Q8. How would you improve the project?**

I would first evaluate retrieval and answer quality using representative questions and expected evidence. Then I would improve chunking, retrieval, citation validation, and handling of questions that cannot be answered from the document.

**Q9. How do you know whether the answer is correct?**

The source passages help a user inspect the evidence, but they are not sufficient by themselves to prove correctness. I would create an evaluation dataset and measure retrieval quality, answer correctness, and whether statements are supported by the retrieved text.

**Q10. Why use a local language model?**

A local model can reduce dependence on hosted generation APIs and allow document processing without sending prompts to an external model provider. However, local processing alone does not guarantee complete privacy; application storage, logs, and other services must also be considered.

---

## 18. Common Issues

### Issue 1: Streamlit Command Not Found

Try:

```powershell
python -m streamlit run app.py
```

Confirm that the virtual environment is active and Streamlit is installed.

### Issue 2: Ollama Model Not Found

Check installed models:

```powershell
ollama list
```

Download the configured model if necessary:

```powershell
ollama pull llama3.2:3b
```

Verify that the model name in your code matches the installed model.

### Issue 3: Import Errors

Run commands from the project root directory.

Check that package imports match the current structure, for example:

```python
from src.vector_stores import load_knowledge_base
```

Use the actual function names defined in your files.

### Issue 4: PDF Produces No Text

The file may be scanned, encrypted, damaged, or contain no extractable text. OCR may be necessary for scanned documents.

### Issue 5: Incorrect Answers

Inspect the retrieved passages first. If the passages are irrelevant, investigate chunking, embeddings, and retrieval. If the passages are relevant but the answer is wrong, investigate prompt construction and model behavior.

### Issue 6: Slow First Run

The embedding model may need to download its files on the first run. Local language-model inference may also take time depending on hardware.

### Issue 7: HTML Appears as Text in the Interface

Use Streamlit's native components where possible. If literal HTML appears in the interface, inspect the relevant code and confirm that old HTML-based UI code is not still being executed.

---

## 19. Disclaimer

DocuMind AI is an educational and experimental AI application. Its answers may be incomplete, inaccurate, or unsupported by the source document.

Users should verify important information against the original document. The application should not be used as the sole basis for legal, medical, financial, or other high-impact decisions.

---

## Project Summary

**DocuMind AI** demonstrates how document processing, pretrained embeddings, vector search, and local language-model generation can be combined into a practical Retrieval-Augmented Generation application.

The project's next stage is to strengthen end-to-end testing, evaluate retrieval and answer quality, improve document isolation, and document reproducible installation steps.

**Built with Python, Streamlit, Sentence Transformers, FAISS, PyMuPDF, and Ollama.**
