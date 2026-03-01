
import streamlit as st
import os
import requests
import time
import logging
from dotenv import load_dotenv

# Import Supabase helpers
try:
    from supabase_client import create_chat_session, get_all_chat_sessions, delete_chat_session, get_chat_history, save_message
except ImportError:
    # Fallback if supabase client fails
    def create_chat_session(*a, **k): return None
    def get_all_chat_sessions(*a, **k): return []
    def delete_chat_session(*a, **k): return False
    def get_chat_history(*a, **k): return []
    def save_message(*a, **k): return False

# --- Configuration & Setup ---
load_dotenv()
st.set_page_config(
    page_title="Pharma AI Assistant (Optimized)",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d333b;
    }
    
    /* Chat Message Styles */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stChatMessage[data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #00C6FF, #0072FF);
    }
    
    .stChatMessage[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }

    /* Input Box Styling */
    .stTextInput > div > div > input {
        background-color: #21262d;
        color: #e0e0e0;
        border-radius: 20px;
        border: 1px solid #30363d;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_files():
    try:
        response = requests.get(f"{API_URL}/files")
        if response.status_code == 200:
            return response.json().get("files", [])
    except:
        pass
    return []

# --- Sidebar: Knowledge Base ---
with st.sidebar:
    st.title("📂 Knowledge Base")
    
    # System Status
    api_online = check_api_health()
    if api_online:
        st.success("Backend API: Online 🟢")
    else:
        st.error("Backend API: Offline 🔴 (or still starting...)")
        st.caption("If you used `start_app.bat`, the backend might still be loading its AI models. Wait ~30-40 seconds and click **Refresh**.")

    # --- File Management ---
    st.markdown("### 📚 Managed Documents")
    
    if api_online:
        current_files = get_files()
        
        if current_files:
            for file in current_files:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.caption(f"📄 {file}")
                with col2:
                    if st.button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
                        try:
                            # Use source filename to delete (assuming source is filename for now)
                            # Or we might need to be careful if source is full path
                            # The API expects filename/source string
                            res = requests.delete(f"{API_URL}/files/{file}")
                            if res.status_code == 200:
                                st.success("Deleted")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No documents found.")

        st.markdown("---")
        st.markdown("### ➕ Add New Document")
        
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Drag and drop your PDF file here.")
        
        if uploaded_file is not None:
            # Check if it's already in the DB
            current_files = get_files()
            if uploaded_file.name in current_files:
                st.info(f"✅ **{uploaded_file.name}** is already in the Knowledge Base.")
            elif st.session_state.processed_file != uploaded_file.name:
                with st.spinner(f"🚀 Processing **{uploaded_file.name}**... (This takes 30-60 secs)"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        # Increase timeout since processing is synchronous now
                        response = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                        
                        if response.status_code == 200:
                            st.session_state.processed_file = uploaded_file.name
                            st.success(f"✅ Subscribed & processed: **{uploaded_file.name}**")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Processing failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error communicating with API: {e}")
        
        if st.button("Refresh File List"):
            st.rerun()

    else:
        st.warning("Connect to API to manage files.")

    st.markdown("---")
    
    # --- Chat History Management ---
    st.markdown("### 💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.messages = []
        st.rerun()
        
    sessions = get_all_chat_sessions()
    if sessions:
        for s in sessions:
            title = s.get('title') or "Chat"
            if len(title) > 20: 
                title = title[:17] + "..."
                
            cols = st.columns([0.75, 0.25])
            with cols[0]:
                if st.button(title, key=f"load_{s['id']}", use_container_width=True, help="Load this chat"):
                    st.session_state.current_session_id = s['id']
                    history = get_chat_history(s['id'])
                    st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
                    st.rerun()
            with cols[1]:
                if st.button("🗑️", key=f"del_{s['id']}", help="Delete this chat"):
                    delete_chat_session(s['id'])
                    if st.session_state.current_session_id == s['id']:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.info("No past chats found.")
    
    st.markdown("---")
    st.caption("Powered by Docling, ChromaDB & Groq")

# --- Main Chat Interface ---
st.title("💊 Pharma AI Assistant")
st.markdown("Ask detailed questions about your uploaded documents. I can extract insights from texts, tables, and images.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    # Auto-create session if none exists
    if st.session_state.current_session_id is None:
        title_snippet = prompt[:30] + "..." if len(prompt) > 30 else prompt
        new_id = create_chat_session(title=title_snippet)
        st.session_state.current_session_id = new_id

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.current_session_id:
        save_message(st.session_state.current_session_id, "user", prompt)
        
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if api_online:
        with st.chat_message("assistant"):
            try:
                with st.spinner("🧠 Thinking & Verifying (API)..."):
                    response = requests.post(f"{API_URL}/query", json={"question": prompt})
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer received.")
                        images = data.get("images", [])
                        
                        st.markdown(answer)
                        
                        # Save response to history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        if st.session_state.current_session_id:
                            save_message(st.session_state.current_session_id, "assistant", answer)
                        
                        # Display Images
                        if images:
                            with st.expander("🖼️ Relevant Images / Figures", expanded=True):
                                cols = st.columns(4)
                                for idx, img_path in enumerate(sorted(images)):
                                    with cols[idx % 4]:
                                        if os.path.exists(img_path):
                                             st.image(img_path, caption="Figure", width=150)
                                        else:
                                             # Try to handle relative path if run from different dir
                                             # Assuming img_path is relative to app root
                                             st.caption(f"Image: {os.path.basename(img_path)}")
                    else:
                        error_msg = f"API Error: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            except Exception as e:
                st.error(f"Connection Error: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("Backend API is unreachable.")
