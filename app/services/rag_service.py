from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


def create_documents(pages: list, filename: str):
    """
    Convert extracted PDF pages into smaller chunks.

    Each chunk keeps metadata containing:
    - document name
    - page number
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    documents = []

    for page in pages:

        chunks = text_splitter.create_documents(
            [page["text"]],
            metadatas=[
                {
                    "document": filename,
                    "page": page["page_number"]
                }
            ]
        )

        documents.extend(chunks)

    return documents


def create_vector_store(pages: list, filename: str):
    """
    Create a FAISS vector store from a document.
    """

    documents = create_documents(
        pages,
        filename
    )

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(
        documents,
        embedding=embeddings
    )

    return vector_store


def add_to_vector_store(
    vector_store,
    pages: list,
    filename: str
):
    """
    Add a new document to an existing FAISS vector store.
    """

    documents = create_documents(
        pages,
        filename
    )

    vector_store.add_documents(documents)

    return vector_store


def ask_question(
    vector_store,
    question: str,
    chat_history: list | None = None,
    document: str | None = None
):
    """
    Retrieve relevant chunks and generate an answer.

    If a document is specified, retrieval is restricted
    to that document.
    """

    if chat_history is None:
        chat_history = []

    # --------------------------------------------------
    # Retrieval configuration
    # --------------------------------------------------

    search_kwargs = {}

    if document:
        search_kwargs["filter"] = {
            "document": document
        }

    relevant_docs = vector_store.similarity_search(
        question,
        k=4,
        **search_kwargs
    )

    # --------------------------------------------------
    # No relevant information
    # --------------------------------------------------

    if not relevant_docs:

        return {
            "answer": (
                "I could not find the answer in the document."
            ),
            "sources": []
        }

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context_parts = []

    for doc in relevant_docs:

        document_name = doc.metadata.get(
            "document",
            "Unknown document"
        )

        page_number = doc.metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
Document: {document_name}
Page: {page_number}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    # --------------------------------------------------
    # Build conversation history
    # --------------------------------------------------

    conversation = ""

    for message in chat_history:

        conversation += (
            f"User: {message['question']}\n"
            f"Assistant: {message['answer']}\n\n"
        )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    prompt = f"""
You are IntelliDocs AI, an AI-powered document
question-answering assistant.

Your job is to answer questions using ONLY the
information contained in the provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer cannot be found in the provided
   context, say:
   "I could not find the answer in the document."
4. Keep the answer clear and concise.
5. Use conversation history only when necessary
   to understand references such as:
   "the first one", "that API", or "what about it?"
6. Do not mention information that is not supported
   by the document context.

Conversation history:
{conversation}

Document context:
{context}

User question:
{question}
"""

    response = llm.invoke(prompt)

    # --------------------------------------------------
    # Build source list
    # --------------------------------------------------

    sources = []

    for doc in relevant_docs:

        source = {
            "document": doc.metadata.get("document"),
            "page": doc.metadata.get("page")
        }

        if source not in sources:
            sources.append(source)

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {
        "answer": response.content,
        "sources": sources
    }