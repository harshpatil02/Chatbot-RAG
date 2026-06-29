from flask import Flask, render_template, request, jsonify
import logging
import os
import psycopg2
from dotenv import load_dotenv
from google import genai
from google.genai import types
import ingest

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize the modern Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    """Converts the user's question into a compressed 768-dimension vector"""
    logger.info("Embedding request received for user input (%d characters).", len(text))
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={"output_dimensionality": 768}
    )
    embedding = response.embeddings[0].values
    logger.info("Embedding generated (%d dimensions).", len(embedding))
    return embedding


def get_db_connection():
    """Return a new Postgres connection from environment variables."""
    logger.info("Connecting to PostgreSQL database %s at host %s.", os.getenv("DB_NAME"), os.getenv("DB_HOST"))
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


def search_vector_db(query_vector, limit=3):
    """Queries Postgres using pgvector to find the closest context chunks"""
    logger.info("Searching vector database with limit %s.", limit)
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
    logger.info("Retrieved %d matching context chunks from the database.", len(results))
    return [row[0] for row in results]

def store_conversation(user_question, response_text):
    """Store the user question and model response in the conversation table."""
    try:
        logger.info("Storing conversation record to DB.")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversation (user_query, response)
            VALUES (%s, %s);
            """,
            (user_question, response_text)
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Conversation record saved.")
    except Exception as e:
        logger.warning("Could not save conversation: %s", e)


def ask_rag(user_question):
    """Main RAG pipeline: embed question, retrieve context, and generate answer"""
    logger.info("User question received: %s", user_question)
    
    try:
        # 1. Vectorize user question
        query_vector = get_embedding(user_question)
        
        # 2. Retrieve matched context chunks from Postgres
        matched_contexts = search_vector_db(query_vector, limit=3)
        
        # Combine retrieved text chunks
        context_text = "\n---\n".join(matched_contexts)
        logger.info("Prepared context text for generation (length %d chars).", len(context_text))
        
        # 3. Construct prompts
        system_prompt = (
            "You are an expert AI Assistant. Use the provided context fragments to answer "
            "the user's question accurately. If the answer cannot be derived from the context, "
            "state that you do not know based on the provided information. Do not make things up."
        )
        
        user_prompt = f"Context Information:\n{context_text}\n\nQuestion: {user_question}"
        logger.info("Constructed prompt for Gemini generation.")
        
        # 4. Generate the final answer using Gemini
        model = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(temperature=0.2)
        )
        
        logger.info("Generated response from Gemini (length %d chars).", len(model.text or ""))
        return model.text
    except Exception as e:
        logger.error("Error processing question: %s", e)
        return f"Error processing your question: {str(e)}"


@app.route('/api/refresh', methods=['POST'])
def refresh_embeddings():
    """Refresh vector store: delete old embeddings and re-ingest documents.

    POST JSON options:
      - mode: 'all' (default) or 'files'
      - filenames: list of filenames to refresh when mode=='files'
    """
    try:
        payload = request.get_json(silent=True) or {}
        mode = payload.get('mode', 'all')
        filenames = payload.get('filenames') or []
        logger.info("Refresh request received with mode=%s, filenames=%s", mode, filenames)

        conn = get_db_connection()
        cursor = conn.cursor()

        if mode == 'files' and filenames:
            logger.info("Deleting embeddings for specified files.")
            deleted_count = 0
            for fname in filenames:
                cursor.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE (metadata::json->>'source') = %s
                       OR (metadata::json->>'source') LIKE %s;
                    """,
                    (fname, f"%{fname}%")
                )
                deleted_count += cursor.rowcount
            logger.info("Deleted %d embedding rows for specified files.", deleted_count)
        else:
            logger.info("Deleting all embeddings from document_chunks.")
            cursor.execute("DELETE FROM document_chunks;")
            logger.info("Deleted all rows from document_chunks.")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("Starting re-ingestion of data/ folder.")
        ingest.ingest_documents()
        logger.info("Re-ingestion complete.")

        return jsonify({'success': True, 'message': 'Embeddings refreshed'})
    except Exception as e:
        logger.error("Refresh failed: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the chatbot HTML page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint to handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        logger.info("Chat request received.")
        
        if not user_message:
            logger.warning("Chat request failed: empty message.")
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get RAG response
        response = ask_rag(user_message)
        
        # Save conversation record
        store_conversation(user_message, response)
        
        logger.info("Returning chat response.")
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        logger.error("Chat API error: %s", e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
