# RAG Application with Pinecone and Google Gemini

A Retrieval-Augmented Generation (RAG) application that combines Pinecone vector database with Google Gemini LLM to provide intelligent question-answering capabilities over PDF documents.

## Features

- **PDF Document Processing**: Automatically loads and processes PDF files from a directory
- **Smart Document Chunking**: Splits documents into optimal chunks for better retrieval
- **Vector Search**: Uses Pinecone with LLaMA embeddings for semantic search
- **AI-Powered Answers**: Leverages Google Gemini 2.0 Flash for generating contextual answers
- **Batch Processing**: Efficiently handles large document collections
- **Environment-Safe**: Uses environment variables for API key management

## Architecture

```
PDF Documents → Document Chunking → Pinecone Vector DB → Query Processing → Gemini LLM → Answer
```

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Gemini API access
- Pinecone account and API key
- PDF documents to process

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd rag-pinecone-gemini
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   ```
   Edit `.env` file and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Getting API Keys

### Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Pinecone API Key
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new project
3. Go to API Keys section
4. Copy your API key to your `.env` file

## Usage

1. **Prepare your documents**
   - Create a directory for your PDF files
   - Place your PDF documents in the directory

2. **Configure the application**
   - Edit the `DOCUMENTS_DIRECTORY` variable in `main.py`
   - Modify the `QUERY` variable to ask your question

3. **Run the application**
   ```bash
   python main.py
   ```

## Configuration Options

### Document Processing
- `chunk_size`: Size of each document chunk (default: 800)
- `chunk_overlap`: Overlap between chunks (default: 50)

### Pinecone Settings
- `index_name`: Name of the Pinecone index (default: "lanchain")
- `cloud`: Cloud provider (default: "aws")
- `region`: AWS region (default: "us-east-1")
- `embedding_model`: Embedding model (default: "llama-text-embed-v2")

### Gemini Settings
- `model`: Gemini model version (default: "gemini-2.0-flash")
- `temperature`: Response creativity (default: 0.5)

## Example Usage

```python
# Example query
QUERY = "How much the agriculture target will be increased by how many crore?"

# Example output
Answer: Based on the documents, the agriculture target will be increased by 2,500 crore rupees for the next fiscal year.
```

## Project Structure

```
rag-pinecone-gemini/
├── main.py              # Main application file
├── .env.template        # Environment variables template
├── .env                 # Your actual environment variables (not in git)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore file
└── documents/          # Directory for your PDF files
```

## Dependencies

- `langchain-community`: Document loaders and utilities
- `langchain-google-genai`: Google Gemini integration
- `langchain`: Core LangChain functionality
- `pinecone-client`: Pinecone vector database client
- `python-dotenv`: Environment variable management
- `pypdf`: PDF processing library

## Troubleshooting

### Common Issues

1. **"No PDF files found"**
   - Check the `DOCUMENTS_DIRECTORY` path
   - Ensure PDF files are in the correct directory

2. **"Missing environment variables"**
   - Verify your `.env` file exists and contains the required keys
   - Check that API keys are valid and active

3. **Pinecone connection errors**
   - Verify your Pinecone API key is correct
   - Check that the index region matches your Pinecone project settings

4. **Google API errors**
   - Ensure your Google API key has Gemini API access enabled
   - Check your API usage limits

### Performance Tips

- Use smaller `chunk_size` for more precise retrieval
- Increase `k` parameter in `retrieve_query()` for more context
- Adjust `temperature` for more creative or conservative responses

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and rotate them regularly
- Use environment variables in production deployments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Pinecone](https://www.pinecone.io/) for vector database services
- [Google AI](https://ai.google/) for the Gemini LLM
- [OpenAI](https://openai.com/) for embedding models

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**Note**: This application processes documents locally and sends queries to external APIs. Ensure you comply with your organization's data handling policies when using sensitive documents.
