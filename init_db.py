import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_database():
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
    
    # 1. Enable pgvector extension explicitly
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. Create table for storing RAG document chunks
    # Note: text-embedding-004 uses 768 dimensions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        metadata JSONB,
        embedding vector(768)
    );
    """)

    # 3. Create table for storing conversation history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation (
        id SERIAL PRIMARY KEY,
        user_query TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)
    
    conn.commit()
    print("✅ Database initialized and tables created successfully with pgvector!")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    init_database()