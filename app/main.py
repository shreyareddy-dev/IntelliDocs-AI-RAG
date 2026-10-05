import os

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.services.document_processor import (
    extract_pages_from_pdf
)

from app.services.rag_service import (
    create_vector_store,
    add_to_vector_store,
    ask_question
)

from app.models.schemas import (
    QuestionRequest,
    AnswerResponse
)


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="IntelliDocs AI",
    description=(
        "AI-powered document intelligence and "
        "Retrieval-Augmented Generation platform"
    ),
    version="1.0.0"
)


# ============================================================
# Temporary in-memory state
# ============================================================

# FAISS vector store
vector_store = None

# Uploaded document names
uploaded_documents = []

# Conversation history
chat_history = []


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():

    if not os.getenv("OPENAI_API_KEY"):

        print(
            "WARNING: OPENAI_API_KEY was not found. "
            "Add it to your .env file."
        )

    else:

        print(
            "OPENAI_API_KEY loaded successfully."
        )


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "IntelliDocs AI API is running"
    }


# ============================================================
# Upload document
# ============================================================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload a PDF and add it to the knowledge base.
    """

    global vector_store

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # --------------------------------------------------------
    # Validate API key
    # --------------------------------------------------------

    if not os.getenv("OPENAI_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Add it to the .env file."
            )
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # --------------------------------------------------------
    # Extract PDF pages
    # --------------------------------------------------------

    try:

        pages = extract_pages_from_pdf(
            file_bytes
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Failed to read PDF: {str(e)}"
        )

    # --------------------------------------------------------
    # Check extracted content
    # --------------------------------------------------------

    if not pages:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract text from this PDF. "
                "The PDF may contain scanned images instead "
                "of selectable text."
            )
        )

    # --------------------------------------------------------
    # Create / update vector store
    # --------------------------------------------------------

    try:

        if vector_store is None:

            vector_store = create_vector_store(
                pages,
                file.filename
            )

        else:

            vector_store = add_to_vector_store(
                vector_store,
                pages,
                file.filename
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to create embeddings: {str(e)}"
        )

    # --------------------------------------------------------
    # Track uploaded document
    # --------------------------------------------------------

    if file.filename not in uploaded_documents:

        uploaded_documents.append(
            file.filename
        )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "pages": len(pages),
        "message": "Document added to knowledge base"
    }


# ============================================================
# List documents
# ============================================================

@app.get("/documents")
def list_documents():
    """
    Return all documents currently loaded
    into the knowledge base.
    """

    return {
        "documents": uploaded_documents,
        "count": len(uploaded_documents)
    }


# ============================================================
# Ask question
# ============================================================

@app.post(
    "/documents/ask",
    response_model=AnswerResponse
)
def ask_document(
    request: QuestionRequest
):
    """
    Ask a question about uploaded documents.

    If a document name is provided, retrieval
    is restricted to that document.
    """

    if vector_store is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload at least one document first."
            )
        )

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # Search knowledge base
    # --------------------------------------------------------

    try:

        result = ask_question(
            vector_store,
            request.question,
            chat_history,
            request.document
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )

    # --------------------------------------------------------
    # Store conversation
    # --------------------------------------------------------

    chat_history.append(
        {
            "question": request.question,
            "answer": result["answer"]
        }
    )

    # --------------------------------------------------------
    # Return answer
    # --------------------------------------------------------

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


# ============================================================
# Clear conversation history
# ============================================================

@app.delete("/documents/history")
def clear_chat_history():
    """
    Clear the current conversation history.
    """

    chat_history.clear()

    return {
        "message": "Conversation history cleared"
    }


# ============================================================
# Get conversation history
# ============================================================

@app.get("/documents/history")
def get_chat_history():
    """
    Return the current conversation history.
    """

    return {
        "history": chat_history,
        "count": len(chat_history)
    }