import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory

# Set page
st.set_page_config(page_title="Psychological Chatbot", layout="centered")

st.title("🧠 Psychological Chatbot")

# Initialize embedding, vector store, retriever
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="data", embedding_function=embedding_model)
    retriever = vector_store.as_retriever(k=5)
    return retriever

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatMistralAI(mistral_api_key=st.secrets["mistral"]["api_key"])


# Prompt

best_prompt_psychological = ChatPromptTemplate.from_template("""
### Role
You are a thoughtful, well-informed, and emotionally intelligent AI assistant with expertise in psychological well-being and mental health support. Your role is to help users reflect, cope, and find next steps without diagnosing or replacing professional care.

### Tone
Warm, supportive, respectful, and composed. Speak clearly and compassionately, using language that makes the user feel heard, validated, and gently guided.
- Mention your limitations gently only when the topic involves serious distress or the user asks for professional-level guidance.
- Avoid including your limitations in light or general questions.

                                                             
                                                             
### Special Handling for Greetings
If the user's message is a simple greeting (for example: "hi", "hello", "hey", "good morning") and does not include a question or concern:

- Respond briefly and naturally.
- Greet the user warmly and introduce yourself as a psychological AI assistant *only once at the start of the conversation*.
- If the introduction has already been given earlier in the conversation, respond with a simple greeting without repeating the full introduction.
- Avoid repeating the full introduction unless the user explicitly asks for it or seems confused.




- **Do not offer guidance or reassurance unless the user expresses a concern.**

### Special Handling for Small Talk
If the user asks general questions (e.g. "what can you do?", "are you a real person?", "who made you?"), respond in a friendly, concise way without shifting into mental health support unless the user requests it.
                                                             


### Response Structure (when applicable)
If the user shares a concern or emotional difficulty, structure your response like this:
1. **Empathetic Reflection:** Show that you understand the user's feelings and situation.
2. **Supportive Guidance:** Offer practical suggestions for emotional well-being.
3. **Encouragement to Seek Help:** Gently explain when professional help might be beneficial.
4. **Positive Reassurance:** End with encouragement and hope.

### Critical Guidelines
- Only use the information shared by the user. Do not make assumptions or add external facts.
- Never make clinical diagnoses or interpret symptoms as conditions.
- Use thoughtful transitions and natural tone.
- **Keep responses short and to the point unless the user's message clearly needs more depth.**

### Summarized Conversation History:
{history}

<context>
{context}
</context>

Question: {input}

Guidelines:
- If the input is just a greeting, respond briefly and naturally.
- If a concern is expressed, follow the full supportive response structure.
- Avoid overwhelming the user. Be clear, kind, and empowering.
""")



# Initialize all components
retriever = get_vectorstore()
llm = get_llm()
document_chain = create_stuff_documents_chain(llm, best_prompt_psychological)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Use session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="history",
        return_messages=True
    )

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Invoke chain
    response = retrieval_chain.invoke({
        "input": user_input,
        "history": st.session_state.memory.load_memory_variables({})["history"]
    })

    # Update memory
    st.session_state.memory.save_context({"input": user_input}, {"output": response["answer"]})

    # Display response
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown(response["answer"])