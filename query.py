import logging
import os
import psycopg2
from dotenv import load_dotenv
from google import genai
from google.genai import types # Added to handle proper configuration schemas

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize the modern Gemini Client
client = genai.Client()

def get_embedding(text):
    """Converts the user's question into a compressed 768-dimension vector"""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={"output_dimensionality": 768}
    )
    return response.embeddings[0].values

def search_vector_db(query_vector, limit=3):
    """Queries Postgres using pgvector to find the closest context chunks"""
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
    cursor = conn.cursor()
    
    # '<=>' calculates Cosine Distance for semantic similarity matching
    cursor.execute(
        """
        SELECT content 
        FROM document_chunks 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s;
        """,
        (query_vector, limit)
    )
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [row[0] for row in results]

def ask_rag(user_question):
    logger.info("Received user question: %s", user_question)
    
    # 1. Vectorize the user's question
    query_vector = get_embedding(user_question)
    
    # 2. Retrieve matched context chunks from Postgres
    matched_contexts = search_vector_db(query_vector, limit=3)
    logger.info("Matched context count: %d", len(matched_contexts))
    
    # Combine retrieved text chunks to feed to the LLM
    context_text = "\n---\n".join(matched_contexts)
    logger.info("Combined context text length: %d", len(context_text))
    
    # 3. Construct System and User Prompts
    system_prompt = (
        "You are an expert AI Assistant. Use the provided context fragments to answer "
        "the user's question accurately. If the answer cannot be derived from the context, "
        "state that you do not know based on the provided information. Do not make things up."
    )
    
    user_prompt = f"Context Information:\n{context_text}\n\nQuestion: {user_question}"
    logger.info("Prepared prompt for Gemini generation.")
    
    # 4. Generate the final answer using the current mainline model and explicit configuration typing
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2
        )
    )
    
    logger.info("Generated AI response with length %d.", len(response.text or ""))
    print("\n🤖 AI Response:")
    print(response.text)

if __name__ == "__main__":
    query = input("\nWhat would you like to ask your documents? ")
    ask_rag(query)