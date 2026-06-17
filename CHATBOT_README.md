# RAG ChatBot Web Interface

A beautiful and interactive chatbot web interface for your RAG (Retrieval-Augmented Generation) system powered by Google Gemini and PostgreSQL with pgvector.

## Features

✨ **Modern UI** - Clean, responsive design with smooth animations
🚀 **Fast & Reliable** - Built with Flask and Gemini 1.5 Flash
📱 **Mobile Friendly** - Works seamlessly on desktop, tablet, and mobile
🔄 **Real-time Streaming** - Get responses instantly from your knowledge base
🎨 **Beautiful Design** - Purple gradient theme with smooth interactions

## Setup Instructions

### 1. Install 
```bash
pip install -r requirements.txt
```

### 2. Set up your environment
Make sure your `.env` file contains:
```
GEMINI_API_KEY=your_gemini_api_key
DB_HOST=localhost
DB_NAME=demoDB
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432
```

### 3. Initialize your database
Before running the web app, ensure your database is set up:
```bash
python init_db.py
```

### 4. Ingest your documents
Add your `.txt` files to the `data/` folder, then:
```bash
python ingest.py
```

### 5. Run the Flask app
```bash
python app.py
```

The chatbot will be available at: **http://localhost:8080**

## File Structure

```
Rag_project/
├── app.py                 # Flask application with RAG pipeline
├── init_db.py            # Database initialization script
├── ingest.py             # Document ingestion script
├── query.py              # RAG query functions (legacy)
├── .env                  # Environment configuration
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Chatbot web interface
└── data/                 # Folder for .txt documents
```

## How to Use

1. **Prepare Documents**: Add `.txt` files to the `data/` folder
2. **Ingest**: Run `python ingest.py` to process documents into vectors
3. **Chat**: Open http://localhost:5000 and start asking questions
4. **Get Answers**: The bot retrieves relevant context and answers using Gemini

## Technology Stack

- **Backend**: Flask + Python
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Gemini Embedding Model 2 (768 dimensions)
- **Vector Database**: PostgreSQL + pgvector
- **Frontend**: HTML5 + CSS3 + JavaScript

## Troubleshooting

**Issue**: Port 5000 already in use
- Solution: Modify `app.py` to use a different port:
  ```python
  app.run(debug=True, host='127.0.0.1', port=5001)
  ```

**Issue**: Database connection error
- Ensure PostgreSQL is running and `.env` credentials are correct

**Issue**: No documents in knowledge base
- Make sure you've added `.txt` files to `data/` and run `python ingest.py`

## Notes

- The embedding model outputs 768-dimensional vectors
- Responses use temperature=0.2 for consistent, factual answers
- The chatbot retrieves top 3 most relevant document chunks for context
