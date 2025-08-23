import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from prompts import prompt_template

load_dotenv()

# Load Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load vectorstore
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5}
)

# Model (Claude via Bedrock)
model = ChatBedrock(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_kwargs={"temperature": 0, "max_tokens": 512}
)

# Memory
memory = ConversationBufferMemory(return_messages=True)

# ---- Formatting Layer ----
def format_response(resp: str) -> str:
    resp = resp.strip()

    # Short factual answers → keep concise
    if len(resp.split()) < 25:
        return resp

    # For longer responses → split into readable paragraphs
    sentences = resp.split(". ")
    formatted = []
    current = []

    for s in sentences:
        current.append(s.strip())
        if len(current) >= 2:  # group 2 sentences together
            formatted.append(". ".join(current) + ".")
            current = []
    if current:
        formatted.append(". ".join(current) + ".")

    return "\n\n".join(formatted)


# Pipeline
def chatbot(query: str):
    retrieved_docs = retriever.get_relevant_documents(query)

    if not retrieved_docs:
        context = " "
    else:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt_template.format(question=query, context=context)
    response = model.invoke(final_prompt)

    memory.save_context({"human": query}, {"ai": str(response.content)})

    return format_response(str(response.content))


if __name__ == "__main__":
    print("🤖 Chatbot ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chatbot(user_input)
        print(f"Bot: {answer}\n")
