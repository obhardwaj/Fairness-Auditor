from langchain_groq import ChatGroq
from app.core.config import settings


def get_llm(temperature: float = 0.2):
    """
    Shared Groq client for all agent nodes. Low temperature by default since
    these are reasoning/classification tasks (which metric applies, is there
    a contradiction, which mitigation to pick) — not creative writing, except
    for the final report_node, which can request a higher temperature call
    directly if a more natural tone is wanted there.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
    )