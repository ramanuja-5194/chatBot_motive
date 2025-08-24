import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from prompts import prompt_template  # <-- use the unified prompt

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-roberta-large-v1"
)

# Load Vectorstore
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

# LLM model
model = ChatBedrock(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_kwargs={"temperature": 0.2, "max_tokens": 512}
)

# Memory
memory = ConversationBufferMemory(return_messages=True)


def chatbot(query: str):
    """Main chatbot logic."""
    # Retrieve past conversation context
    # past_context = "\n".join(
    #     [f"{m.type.upper()}: {str(m.content)}" for m in memory.chat_memory.messages]
    # )
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        context = "I couldn’t find any relevant information."
    else:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Use unified prompt template
    prompt = prompt_template.format_messages(
        question=query,
        context=context
    )

    # Get answer from model
    final_answer = model.invoke(prompt).content

    # Save conversation
    memory.save_context({"human": query}, {"ai": str(final_answer)})

    with open("conversation.md", "a", encoding="utf-8") as f:
        f.write(f"**HUMAN:** {query}\n\n")
        f.write(f"**AI:** {final_answer}\n\n")
        f.write("---\n")

    return final_answer


if __name__ == "__main__":
    print("🤖 Chatbot ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = chatbot(user_input)
        print(f"Bot: {answer}\n")
