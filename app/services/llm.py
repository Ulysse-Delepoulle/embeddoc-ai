from app.config import get_settings
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage, AIMessage


def answer(question: str, chunks: list, low_confidence: bool, history: list[dict]) -> str:
    if low_confidence:
        return "I couldn't find reliable information in the provided datasheets."

    context = "\n\n".join(
        f"[Document: {c['document_name']} | Page: {c['page_num']}]\n{c['text']}"
        for c in chunks
    )

    settings = get_settings()
    chat = ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
    )

    messages = [
        SystemMessage(content=(
            "You are a technical assistant specialized in embedded systems datasheets. "
            "Answer strictly from the provided context. "
            "Always cite your sources using the document name and page number. "
            "If the context does not contain enough information to answer, say so clearly."
        ))
    ]

    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"))

    response = chat.invoke(messages)
    return response.content
