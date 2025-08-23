import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from prompts import prompt_template

load_dotenv()

# Load Hugging Face embeddings (instead of Titan)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # you can change to any HF embedding model
)

# Load vectorstore
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5}
)

# Model (Claude via Bedrock)
model = ChatBedrock(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_kwargs={"temperature": 0, "max_tokens": 512}
)

# Memory
memory = ConversationBufferMemory(return_messages=True)

# Pipeline
def chatbot(query: str):
    retrieved_docs = retriever.get_relevant_documents(query)
    if not retrieved_docs:
        context = "No relevant information found in the documents."
    else:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt_template.format(question=query, context=context)

    response = model.invoke(final_prompt)
    memory.save_context({"human": query}, {"ai": str(response.content)})
    return str(response.content)

if __name__ == "__main__":
    print("🤖 Chatbot ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chatbot(user_input)
        print(f"Bot: {answer}\n")
