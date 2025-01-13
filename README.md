# General Document Analysis System

[中文文档](README_CN.md)

A powerful document analysis system built with LangChain that supports multiple document formats, asynchronous processing, and multi-knowledge base search.

## Features

- **Multiple Document Format Support**
  - PDF files
  - Word documents (.docx)
  - Text files (.txt)
  - Markdown files (.md)
  - Maximum file size: 30MB

- **Asynchronous Document Processing**
  - Asynchronous document loading and vectorization
  - Batch processing for large documents
  - Memory-efficient processing with automatic garbage collection
  - Progress tracking and error recovery

- **Advanced Search Capabilities**
  - Multi-knowledge base concurrent search
  - Context-aware retrieval
  - Reranking for better result relevance
  - Streaming responses for better user experience

- **LangChain Technologies Used**
  - Document Loaders: PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
  - Text Splitters: RecursiveCharacterTextSplitter
  - Vector Stores: FAISS
  - Embeddings: OpenAI Embeddings
  - Retrievers: 
    - ContextualCompressionRetriever
    - MultiVectorRetriever
    - EnsembleRetriever
    - BM25Retriever
  - Rerankers: CohereRerank
  - LLMs: ChatOpenAI
  - Indexes:
    - VectorStoreIndexWrapper
    - VectorStoreIndexCreator
    - Document Index Management

## Evaluation with Ragas

The system includes comprehensive evaluation using Ragas metrics:
- **Context Relevancy**: Evaluates the relevance of retrieved documents
- **Answer Faithfulness**: Measures how well answers align with provided context
- **Answer Relevancy**: Assesses answer relevance to questions
- **Context Precision**: Evaluates the precision of retrieved context
- **Context Recall**: Measures the completeness of retrieved information

Evaluation results help optimize:
- Document chunking strategies
- Retrieval methods
- Answer generation quality
- Overall system performance

### Running Evaluations
```bash
# Prepare test data
mkdir -p ragas_data/your_document_name
# Create test_data_2.csv with columns: question, ground_truth

# Run evaluation
python ragas_test.py
```

The evaluation results will be saved in:
- `ragas_data/your_document_name/evaluate_data_4.csv`

Example test data format (test_data_2.csv):
```csv
question,ground_truth
"What is the hiring process?","The hiring process includes department request, HR verification, and GM approval."
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/general_document_analysis_system.git
cd general_document_analysis_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Upload documents through the web interface (http://localhost:7860)

3. Ask questions about your documents

## Technical Details

### Asynchronous Processing
- Uses Python's asyncio for concurrent operations
- Implements ThreadPoolExecutor for CPU-bound tasks
- Batch processing with configurable batch sizes
- Automatic memory management and garbage collection

### Vector Storage
- FAISS for efficient similarity search
- Persistent storage with automatic recovery
- Incremental updates for large documents

### Retrieval System
- Multi-stage retrieval pipeline
- Hybrid search combining dense and sparse retrievers
- Context compression for better relevance
- Reranking for improved result quality

### Error Handling
- Comprehensive error recovery
- Failed batch tracking
- Automatic retries for transient failures
- Detailed logging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License - see the LICENSE file for details.