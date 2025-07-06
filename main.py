# import Libraries
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

## Read the document
def read_doc(directory):
    """
    Load PDF documents from a directory
    
    Args:
        directory (str): Path to directory containing PDF files
        
    Returns:
        list: List of loaded documents
    """
    try:
        file_loader = PyPDFDirectoryLoader(directory)
        documents = file_loader.load()
        if not documents:
            print(f"No PDF files found in directory: {directory}")
        return documents
    except Exception as e:
        print(f"Error loading documents from {directory}: {e}")
        return []

## Divide the docs into chunks
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """
    Split documents into smaller chunks for processing
    
    Args:
        docs (list): List of documents to chunk
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of chunked documents
    """
    if not docs:
        print("No documents to chunk.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

## Cosine Similarity Retrieve Results from VectorDB
def retrieve_query(query, k=2):
    """
    Query the Pinecone index to retrieve relevant documents
    
    Args:
        query (str): Search query
        k (int): Number of top results to return
        
    Returns:
        list: List of relevant document hits
    """
    results = index.query(
        namespace="default",
        query={
            "top_k": k,
            "inputs": {
                "text": query
            }
        }
    )
    return [hit for hit in results['result']['hits']]

## Search answers from VectorDB
def retrieve_answers(query):
    """
    Retrieve and generate answers using the QA chain
    
    Args:
        query (str): User question
        
    Returns:
        str: Generated answer
    """
    doc_search = retrieve_query(query)
    print(f"Retrieved documents: {doc_search}")
    response = chain.invoke({"input_documents": doc_search, "question": query})
    return response

def setup_pinecone_index(documents):
    """
    Set up Pinecone index and upsert documents
    
    Args:
        documents (list): List of chunked documents
        
    Returns:
        Pinecone.Index: Configured Pinecone index
    """
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "lanchain"

        # Delete existing index if it exists to ensure correct configuration
        if index_name in pc.list_indexes().names():
            print(f"Deleting existing index '{index_name}'...")
            pc.delete_index(index_name)
            time.sleep(10)  # Wait for deletion

        # Create new index with integrated embedding
        print(f"Creating index '{index_name}' with integrated embedding...")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
        time.sleep(10)  # Wait for index creation
        index = pc.Index(index_name)

        # Prepare text data for upsert (using integrated embedding)
        records = [
            {
                "_id": f"doc_{i}",
                "chunk_text": doc.page_content,
                "category": "budget"  # Assign a default category
            } for i, doc in enumerate(documents)
        ]

        # Upsert data in smaller batches
        batch_size = 50
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            print(f"Upserting batch {i//batch_size + 1} with {len(batch)} records")
            index.upsert_records(namespace="default", records=batch)
        print(f"Upserted {len(documents)} documents into index '{index_name}'.")

        # View index stats
        stats = index.describe_index_stats()
        print("Index stats:", stats)
        
        return index

    except Exception as e:
        print(f"Failed to initialize Pinecone or upsert data: {e}")
        print("Please verify the API key and ensure the 'lanchain' index exists in the us-east-1 region.")
        raise

def main():
    """
    Main function to execute the RAG pipeline
    """
    # Configuration
    DOCUMENTS_DIRECTORY = "F:\\DBLear\\documents"  # Change this to your documents directory
    QUERY = "How much the agriculture target will be increased by how many crore?"
    
    # Read documents
    print("Loading documents...")
    doc = read_doc(DOCUMENTS_DIRECTORY)
    print(f"Number of documents loaded: {len(doc)}")

    if not doc:
        print("No documents found. Please check the directory path.")
        return

    # Chunk documents
    print("Chunking documents...")
    documents = chunk_data(docs=doc)
    print(f"Number of documents after chunking: {len(documents)}")

    # Set up Pinecone index
    print("Setting up Pinecone index...")
    global index
    index = setup_pinecone_index(documents)

    # Initialize Gemini LLM and QA chain
    print("Initializing LLM and QA chain...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=os.environ["GOOGLE_API_KEY"], 
        temperature=0.5
    )
    global chain
    chain = create_stuff_documents_chain(llm, prompt=None)  # Using default prompt for simplicity

    # Query and retrieve answer
    print(f"Processing query: {QUERY}")
    answer = retrieve_answers(QUERY)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with your API keys.")
        exit(1)
    
    main()
