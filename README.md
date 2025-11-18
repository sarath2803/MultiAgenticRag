# Vision-Enhanced Multi-Agent RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that combines intent classification, multimodal retrieval, vision processing, and LLM reasoning in a multi-agent architecture. The system can process PDF documents, extract and analyze images, and provide intelligent answers based on user queries.

## 🎯 Features

- *Intent-Aware Processing*: Automatically classifies queries into four categories (fact, analysis, summary, visual)
- *Multimodal Retrieval*: Handles both text documents and images from PDFs
- *Vision Processing*: Extracts and analyzes images using BLIP captioning and OCR
- *Conversational Memory*: Maintains context across multiple interactions
- *Streamlit UI*: Beautiful, interactive web interface for querying the system
- *ChromaDB Integration*: Efficient vector storage and retrieval using ChromaDB

## 🏗 Architecture

The system consists of six specialized agents working together:

### 1. *Intent Agent* (agents/intent_agent.py)
- Classifies user queries into intent categories: fact, analysis, summary, visual
- Uses a custom PyTorch neural network or rule-based fallback
- Leverages Sentence Transformers for embeddings

### 2. *Retrieval Agent* (agents/retrieval_agent.py)
- Ingests PDF documents and images
- Extracts text chunks and images
- Manages ChromaDB vector database
- Retrieves relevant documents based on query embeddings

### 3. *Vision Agent* (agents/vision_agent.py)
- Generates captions for images using BLIP model
- Performs OCR using Tesseract

### 4. *Reasoning Agent* (agents/reasoning_agent.py)
- Uses Mistral-7B LLM (via llama-cpp-python) for answer generation
- Handles different reasoning modes based on intent
- Integrates retrieved context and memory into prompts

### 5. *Memory Agent* (agents/memory_agent.py)
- Maintains short-term conversation history
- Stores long-term summaries
- Provides context for subsequent queries

### 6. *Controller Agent* (agents/controller_agent.py)
- Orchestrates the entire pipeline
- Routes queries based on intent classification
- Coordinates all agents to produce final answers

## 📁 Project Structure


MULIIAGENTICRAG/
├── agents/
│   ├── controller_agent.py    # Main orchestration agent
│   ├── intent_agent.py         # Intent classification
│   ├── memory_agent.py         # Conversation memory
│   ├── reasoning_agent.py      # LLM reasoning
│   ├── retrieval_agent.py      # RAG retrieval
│   └── vision_agent.py         # Image processing
├── chroma_db/                  # ChromaDB vector database
├── data/                       # Input documents
│   ├── *.pdf                  # PDF documents
│   └── images/                # Extracted images
├── models/
│   └── intent_model/          # Trained intent classifier
│       └── intent_classifier.pt
├── main.py                    # CLI interface
├── streamlit.py               # Streamlit web UI
├── requirements.txt           # Python dependencies
└── README.md                  # This file


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Tesseract OCR installed on your system

### Step 1: Install Tesseract OCR

*macOS:*
bash
brew install tesseract


*Ubuntu/Debian:*
bash
sudo apt-get install tesseract-ocr


*Windows:*
Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Step 2: Install Python Dependencies

bash
pip install -r requirements.txt


### Step 3: Download Mistral Model

The system requires a Mistral-7B model in GGUF format. Download it and place it in the models/mistral/ directory:

bash
mkdir -p models/mistral
# Download mistral-7b-instruct-v0.2.Q4_K_M.gguf to models/mistral/


You can download from [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) or use any compatible Mistral GGUF model.

### Step 4: Prepare Your Data

Place your PDF documents in the data/ directory:

data/
├── Doc1.pdf
├── Doc2.pdf
├── Doc3.pdf
└── images/          # Images will be extracted here automatically


## 💻 Usage

### Option 1: Streamlit Web Interface (Recommended)

Launch the interactive web UI:

bash
streamlit run streamlit.py


The interface will open in your browser at http://localhost:8501.

*Features:*
- Interactive query input
- Real-time intent classification
- Visual display of retrieved documents and images
- Conversation history tracking
- Memory context visualization

### Option 2: Command Line Interface

Run the CLI version:

bash
python main.py


This provides a simple text-based interface for querying the system.

## 📖 Example Queries

### Fact Queries

What is the main topic of the document?
What does page 5 say about predictive analytics?


### Analysis Queries

Compare the results between different documents
Why did the AI use increase inventory management?
Analyze the difference between predictive analytics and custom hospital management?


### Summary Queries

Summarize the key findings
Give me an overview of the document
What are the main points?


### Visual Queries

explain the flowchart of AI in inventory management?
Describe the figure on page 20
show the chart in page 25?


## ⚙ Configuration

### Intent Classification

The intent classifier uses a trained PyTorch model. If the model is not found, it falls back to rule-based classification.

To train a custom intent classifier, use the notebook in models/train_intent.ipynb.

### Retrieval Settings

Adjust retrieval parameters in the code:
- top_k: Number of documents to retrieve (default: 5)
- chunk_chars: Text chunk size for PDFs (default: 900)
- overlap: Overlap between chunks (default: 200)

### Memory Settings

Configure memory in MemoryAgent initialization:
- max_turns: Maximum conversation turns to remember (default: 12)

### Model Paths

Update model paths in the respective agent files:
- *Reasoning Agent*: models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf
- *Intent Model*: models/intent_model/intent_classifier.pt
- *Embedding Model*: all-MiniLM-L6-v2 (Sentence Transformers)

## 🔧 Technical Details

### Intent Types

1. *fact*: Direct factual questions requiring specific information
2. *analysis*: Questions requiring comparison, reasoning, or synthesis
3. *summary*: Requests for overview or key points
4. *visual*: Queries about charts, figures, tables, or images

### Processing Pipeline

1. *Query Input* → User submits query
2. *Intent Classification* → IntentAgent determines query type
3. *Retrieval* → RetrievalAgent fetches relevant documents/images
4. *Vision Processing* → VisionAgent processes images (if visual intent)
5. *Memory Integration* → MemoryAgent provides conversation context
6. *Reasoning* → ReasoningAgent generates answer using LLM
7. *Response* → ControllerAgent returns formatted result

### Vector Database

- *Database*: ChromaDB (persistent storage)
- *Embedding Model*: Sentence Transformers (all-MiniLM-L6-v2)
- *Storage*: Local filesystem (chroma_db/)

## 🐛 Troubleshooting

### Common Issues

*1. Tesseract not found*

Error: TesseractNotFoundError

*Solution*: Install Tesseract OCR (see Installation section)

*2. Mistral model not found*

Error: FileNotFoundError: Mistral model not found

*Solution*: Download and place the Mistral GGUF model in models/mistral/

*3. CUDA out of memory*

Error: RuntimeError: CUDA out of memory

*Solution*: Reduce batch sizes or use CPU mode

*4. BLIP model download fails*

Error: BLIP load failed

*Solution*: The system will fall back to filename-based captions. Check internet connection for model download.

## 📊 Performance Considerations

- *First Run*: Data ingestion may take time depending on document size
- *GPU Acceleration*: Significantly speeds up embeddings and vision processing
- *Memory Usage*: LLM loading requires ~4-8GB RAM for Mistral-7B Q4
- *Storage*: ChromaDB and extracted images can consume significant disk space

## 🔄 Data Ingestion

The system automatically ingests data on first run:
- PDFs are chunked and embedded
- Images are extracted and processed
- OCR text is extracted from images
- All data is stored in ChromaDB

To re-ingest data, delete the chroma_db/ directory and restart.

## 🤝 Contributing

This is a research/development project. Key areas for improvement:
- Enhanced intent classification accuracy
- Better image retrieval and matching
- Improved memory management
- Support for additional document formats
- Performance optimizations

## 📝 License

This project is for research and development purposes.

## 🙏 Acknowledgments

- *Sentence Transformers*: For embeddings
- *ChromaDB*: For vector storage
- *BLIP*: For image captioning
- *Mistral AI*: For the LLM model
- *llama-cpp-python*: For efficient LLM inference
- *Streamlit*: For the web interface

## 📧 Support

For issues or questions, please check:
1. Installation requirements are met
2. Model files are in correct locations
3. Dependencies are properly installed
4. System has sufficient resources (RAM, disk space)

---

*Built with ❤ using Multi-Agent Architecture, RAG, and Vision-Language Models*

