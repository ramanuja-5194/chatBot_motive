from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a helpful, polite, and concise customer support chatbot.
You ONLY answer using information from the provided context.
If the answer is not present, say: "I couldn’t find any relevant details". and provide the closest relavent details."
Never hallucinate. Be polite even for irrelevant or rude questions.
"""

HUMAN_PROMPT = """Customer query:
{question}

Context from documents:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT)
])
