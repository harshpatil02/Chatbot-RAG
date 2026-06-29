import logging
import os
import psycopg2
from json import dumps
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize the modern Gemini Client
# It automatically looks for GEMINI_API_KEY in your .env file
client = genai.Client()

def get_embedding(text):
    """Generates a compressed 768-dimension vector embedding using Gemini"""
    logger.info("Embedding text chunk (%d chars).", len(text))
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={"output_dimensionality": 768}  # Added this configuration line
    )
    embedding = response.embeddings[0].values
    logger.info("Generated embedding vector of length %d.", len(embedding))
    return embedding

def ingest_documents():
    logger.info("Reading documents from data folder...")
    loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
    raw_documents = loader.load()
    logger.info("Loaded %d text documents from data/.", len(raw_documents))
    
    # Also process any Excel files (.xlsx/.xls) in the data folder
    excel_docs = []
    for root, _, files in os.walk("data"):
        for fname in files:
            if fname.lower().endswith(('.xlsx', '.xls')):
                fpath = os.path.join(root, fname)
                logger.info("Reading Excel file: %s", fpath)
                try:
                    df = pd.read_excel(fpath)
                except Exception as e:
                    logger.warning("Skipping %s: failed to read Excel (%s)", fpath, e)
                    continue

                logger.info("Excel file %s loaded with %d rows and %d columns.", fpath, len(df), len(df.columns))
                # For each row, join available columns into a single text blob
                for i, row in df.iterrows():
                    # Convert each cell to string and include column name for context
                    parts = [f"{col}: {row[col]}" for col in df.columns]
                    content = "\n".join(parts)
                    metadata = {"source": fname, "row": int(i)}
                    excel_docs.append(Document(page_content=content, metadata=metadata))

    raw_documents.extend(excel_docs)
    logger.info("Total documents after including Excel: %d", len(raw_documents))

    if not raw_documents:
        logger.warning("No text or Excel files found in the 'data/' folder.")
        return

    logger.info("Splitting %d pages into chunks...", len(raw_documents))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_documents)
    logger.info("Created %d text chunks.", len(chunks))

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()

    logger.info("Generating embeddings and saving to PostgreSQL...")
    for idx, chunk in enumerate(chunks):
        text_content = chunk.page_content
        metadata = dumps(chunk.metadata)
        
        # Generate the 768 vector
        embedding_vector = get_embedding(text_content)
        
        cursor.execute(
            """
            INSERT INTO document_chunks (content, metadata, embedding)
            VALUES (%s, %s, %s);
            """,
            (text_content, metadata, embedding_vector)
        )
        
        if (idx + 1) % 5 == 0 or (idx + 1) == len(chunks):
            logger.info("Saved %d/%d chunks...", idx + 1, len(chunks))

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Ingestion complete! Data successfully vectorized in Postgres.")

if __name__ == "__main__":
    ingest_documents()