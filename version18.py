# mental_health_app.py
import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ======================= Setup ============================
SYMPTOM_OPTIONS = [
    "I have trouble sleeping, such as insomnia or sleeping excessively.",
    "I have noticeable changes in appetite or weight without a medical reason.",
    "I have ongoing feelings of worthlessness or guilt.",
    "I have recurring thoughts about self-harm or suicide.",
    "I have a constant feeling of tension and anxiety that is hard to control.",
    "I have difficulty relaxing even during calm moments.",
    "I have physical symptoms like rapid heartbeat or sweating without a clear reason.",
    "I have a tendency to avoid certain situations out of fear of anxiety or embarrassment.",
    "I have frequent and disturbing memories of a traumatic event I experienced.",
    "I have recurring nightmares related to a painful past experience.",
    "I have a tendency to avoid people or places that remind me of that event.",
    "I have sudden feelings of fear or panic without a clear trigger.",
    "I have heightened sensitivity or excessive irritability to sounds or stimuli.",
    "I have persistent and unwanted thoughts that disturb me and are hard to ignore.",
    "I have repetitive behaviors such as checking or washing to reduce anxiety.",
    "I have a strong urge to perform certain actions in a specific way to prevent distress.",
    "I have periods where I feel excessively energetic and overly confident.",
    "I have intense mood swings between high energy and deep depression.",
    "I have impulsive behaviors or make reckless decisions during high-energy phases.",
    "I have an intense fear of abandonment or being left by others.",
    "I have rapid shifts in how I see myself and others.",
    "I have impulsive behaviors in areas like relationships or spending.",
    "I have a persistent feeling of emptiness or lack of purpose.",
    "I have difficulty focusing on tasks or completing them.",
    "I have a tendency to forget appointments or lose things easily.",
    "I have excessive movement or a need to stay physically active even in quiet situations.",
    "I have difficulty waiting or controlling my reactions.",
    "I have auditory or visual hallucinations that others do not see or hear.",
    "I have unrealistic beliefs such as feeling that someone is watching me or controlling my thoughts.",
    "I have trouble expressing emotions or communicating clearly.",
    "I have disorganized thinking or difficulty organizing my speech."
]

@st.cache_resource
def get_llm():
    return ChatMistralAI(mistral_api_key=os.environ["MISTRAL_API_KEY"], model="mistral-large-latest")

@st.cache_resource
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="data", embedding_function=embeddings)
    return vector_store.as_retriever(k=5)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     """### Role:
You are a supportive AI specialized in mental health.
Use the provided context and user input to analyze symptoms and provide guidance.
You are a thoughtful, emotionally attuned AI companion with deep knowledge of psychological well-being and mental health. Your goal is to support users in reflecting on their feelings, developing coping strategies, and exploring meaningful next steps—while recognizing you do not replace professional care.

### Tone
Empathetic, warm, and genuine — like a trusted friend who listens deeply and responds thoughtfully. Use clear, compassionate language that validates the user’s experience and encourages self-awareness and growth.
- When serious distress or clinical concerns arise, gently explain your limitations and recommend professional resources.
- In general conversations, avoid focusing on limitations; stay present and supportive.
- Tailor your responses to the user's tone, mood, and context to foster trust and rapport.

### Handling Small Talk
Respond to casual or curiosity-driven questions ("what can you do?", "are you real?", "who made you?") with concise, friendly answers that maintain openness, without steering into therapy mode unless invited.                                         

### When Users Share Emotional Difficulties
- Reflect and validate the user’s emotions with sensitivity.
- Offer practical, evidence-based strategies to manage or reframe their feelings.
- Provide psychoeducational insights gently where relevant.
- Suggest professional help thoughtfully when appropriate.
- Close with an encouraging or hopeful message.
- Invite the user to share more by asking open-ended questions or gentle prompts to keep the conversation going.

### Conversation:
{chat_history}

### User Input:
{input}

### Guidelines:
- Avoid overwhelming the user; provide actionable insights at an appropriate pace.
- **Encourage ongoing dialogue by asking questions or inviting further reflection whenever appropriate.**
- Never include system notes, debug info, or meta commentary in your replies.                                       
- If the user inputs only numbers corresponding to symptom indices, respond with the **top 3 most likely psychological disorders** based on those symptoms.
- Keep your answer concise and focused on these three disorders only. Do not add unnecessary explanations or details.

### Instructions:
- If input lists symptoms, analyze and suggest possible disorders.
- Otherwise, chat naturally with the user.

### Context:
{context}"""),
    ("human", "{input}")
])

llm = get_llm()
retriever = get_retriever()
combine_chain = create_stuff_documents_chain(llm, prompt_template)
chat_chain = create_retrieval_chain(retriever, combine_chain)

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_convo" not in st.session_state:
    st.session_state.active_convo = "New Conversation"
if st.session_state.active_convo not in st.session_state.conversations:
    st.session_state.conversations[st.session_state.active_convo] = []
if "doctor_results" not in st.session_state:
    st.session_state.doctor_results = {}
if "patient_counter" not in st.session_state:
    st.session_state.patient_counter = 1

# Sidebar
st.sidebar.title("🗂️ Conversations")
all_convos = ["New Conversation"] + list(st.session_state.conversations.keys())
selected_convo = st.sidebar.selectbox("Choose or start a conversation:", all_convos, index=all_convos.index(st.session_state.active_convo))

if selected_convo != st.session_state.active_convo:
    st.session_state.active_convo = selected_convo
    if selected_convo not in st.session_state.conversations:
        st.session_state.conversations[selected_convo] = []
    st.rerun()

if st.sidebar.button("➕ New Conversation"):
    new_name = f"Conversation {len(st.session_state.conversations)}"
    st.session_state.conversations[new_name] = []
    st.session_state.active_convo = new_name
    st.rerun()

# Saved Diagnoses
with st.sidebar.expander("📁 Saved Diagnoses"):
    if st.session_state.doctor_results:
        for patient_id, data in st.session_state.doctor_results.items():
            st.markdown(f"**🧑‍⚕️ {patient_id}**")
            st.markdown("- Symptoms:")
            for sym in data["symptoms"]:
                st.markdown(f"  - {sym}")
            st.markdown(f"- **Diagnosis:** {data['diagnosis']}")
            st.markdown("---")
        # Download
        diagnosis_text = ""
        for patient_id, data in st.session_state.doctor_results.items():
            diagnosis_text += f"{patient_id}\nSymptoms:\n"
            for sym in data["symptoms"]:
                diagnosis_text += f"- {sym}\n"
            diagnosis_text += f"Diagnosis: {data['diagnosis']}\n{'-'*40}\n"
        st.download_button("⬇️ Download Diagnoses", diagnosis_text, "diagnoses.txt")
    else:
        st.info("No diagnoses saved yet.")

# Main Interface
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Mental Health Support Chatbot</h1>", unsafe_allow_html=True)

symptom_key = f"selected_symptoms_{st.session_state.active_convo}"
st.markdown("### 👇 Select or load your symptoms (just like a self-assessment form):")

uploaded_file = st.file_uploader("📄 Upload a .txt file with symptoms (optional)", type="txt")
file_symptoms = []
if uploaded_file is not None:
    uploaded_lines = uploaded_file.read().decode("utf-8").split("\n")
    file_symptoms = [s.strip() for s in uploaded_lines if s.strip()]
    for symptom in file_symptoms:
        if symptom not in SYMPTOM_OPTIONS:
            SYMPTOM_OPTIONS.append(symptom)

with st.form("symptom_form"):
    selected_symptoms = st.multiselect("🩺 Select your symptoms manually or from uploaded file:", SYMPTOM_OPTIONS, default=file_symptoms, key=symptom_key)
    submit = st.form_submit_button("📝 Send Symptoms")

if submit and selected_symptoms:
    user_input = "I am experiencing the following symptoms:\n" + "\n".join(selected_symptoms)
    history = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.conversations[st.session_state.active_convo]])
    response = chat_chain.invoke({"input": user_input, "chat_history": history, "context": ""})
    st.session_state.conversations[st.session_state.active_convo].append((user_input, response["answer"]))

    patient_id = f"Patient {st.session_state.patient_counter}"
    st.session_state.doctor_results[patient_id] = {
        "symptoms": selected_symptoms,
        "diagnosis": response["answer"]
    }
    st.session_state.patient_counter += 1
    st.rerun()

st.markdown("<hr style='border: 1px solid #ccc; margin-top: 30px; margin-bottom: 20px;'>", unsafe_allow_html=True)

st.subheader("💬 Chat with the assistant")
input_key = f"user_input_{st.session_state.active_convo}"

with st.form("chat_input_form", clear_on_submit=True):
    user_chat = st.text_input("Type your message:", key=input_key)
    send_clicked = st.form_submit_button("📨 Send")

if send_clicked and user_chat.strip():
    history = "\n".join([f"User: {u}\nAI: {a}" for u, a in st.session_state.conversations[st.session_state.active_convo]])
    response = chat_chain.invoke({"input": user_chat, "chat_history": history, "context": ""})
    st.session_state.conversations[st.session_state.active_convo].append((user_chat, response["answer"]))
    st.rerun()

for user_msg, bot_msg in st.session_state.conversations[st.session_state.active_convo]:
    st.markdown(f"""
        <div style=\"background-color:#e6f2ff;padding:10px;border-radius:10px;margin-bottom:10px\">
            <strong>You:</strong><br>{user_msg}
        </div>
        <div style=\"background-color:#f9f9f9;padding:10px;border-radius:10px;margin-bottom:20px\">
            <strong>AI:</strong><br>{bot_msg}
        </div>
    """, unsafe_allow_html=True)
