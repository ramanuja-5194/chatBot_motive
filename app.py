import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Load Vectorstore
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 7}
)

# LLM model
model = ChatBedrock(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_kwargs={"temperature": 0, "max_tokens": 512}
)

# Memory
memory = ConversationBufferMemory(return_messages=True)

# ---------------- Refinement Chains ---------------- #

# Stage 1: Remove boilerplate
clean_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise assistant. Rewrite the answer 
    to remove sentences like 'based on context', 'based on information provided',
    or 'according to the documents'. Provide only the pure useful answer."""),
    ("human", "{draft_answer}")
])

# Stage 2: Conciseness + structured formatting
short_long_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an adaptive rewriter. 
    - If the question is factual and simple → keep the answer within 2 lines.  
    - Otherwise → give a detailed and well-explained answer.  
    - For longer answers, organize the response in a clear, structured format."""),
    ("human", "{cleaned_answer}")
])

# Stage 3: Formatting removed

# Stage 4: Emoji for Yes/No
yesno_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a polite rewriter.
    If the answer contains 'Yes' → add ✅ just after 'YES'.
    If the answer contains 'No' → add ❌ just after 'NO'.
    Otherwise, keep the answer unchanged."""),
    ("human", "{rewritten_answer}")
])

clean_chain = clean_prompt | model | StrOutputParser()
short_long_chain = short_long_prompt | model | StrOutputParser()
yesno_chain = yesno_prompt | model | StrOutputParser()


def refinement_chain(question, draft_answer):
    """Passes the draft answer through multiple refinement stages."""
    cleaned = clean_chain.invoke({"draft_answer": draft_answer})
    rewritten = short_long_chain.invoke({"cleaned_answer": cleaned})
    final_answer = yesno_chain.invoke({"rewritten_answer": rewritten})
    return final_answer


def chatbot(query: str):
    """Main chatbot logic."""
    past_context = "\n".join(
        [f"{m.type.upper()}: {str(m.content)}" for m in memory.chat_memory.messages]
    )
    retrieved_docs = retriever.invoke(query + "\n" + past_context)

    if not retrieved_docs:
        context = "No relevant information found in the documents."
    else:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    base_prompt = f"""Answer the following question strictly using the context.

Question: {query}

Context:
{context}
"""
    # Draft answer from model
    draft_answer = model.invoke(base_prompt).content

    # Refine through chain
    final_answer = refinement_chain(query, draft_answer)

    # Save conversation
    memory.save_context({"human": query}, {"ai": final_answer})

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
