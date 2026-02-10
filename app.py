import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
import os
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import our backend engine
from pharma_rag import IngestionEngine, ChunkingEngine, VectorDatabase, RAGController

# --- Configuration & Setup ---
load_dotenv()
st.set_page_config(
    page_title="Pharma AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_controller" not in st.session_state:
    st.session_state.rag_controller = None
# Code Update Hotfix: Reset controller if it doesn't have the new streaming method
if st.session_state.rag_controller and not hasattr(st.session_state.rag_controller, 'query_stream'):
    st.session_state.rag_controller = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

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
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #38ef7d transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Knowledge Base ---
with st.sidebar:
    st.title("📂 Knowledge Base")
    
    # --- Auto-Load / Initialize ---
    if st.session_state.rag_controller is None:
        if os.path.exists("./chroma_db"):
            try:
                # Initialize DB and Controller without new file
                vector_db = VectorDatabase()
                groq_api_key = os.getenv("GROQ_API_KEY")
                if groq_api_key:
                    st.session_state.rag_controller = RAGController(vector_db, groq_api_key)
                    st.success("✅ Loaded Existing Knowledge Base")
            except Exception as e:
                st.error(f"Failed to load DB: {e}")

    # --- File Management ---
    st.markdown("### 📚 Managed Documents")
    
    # Helper to refresh file list
    def get_file_list():
        if st.session_state.rag_controller:
            return st.session_state.rag_controller.vector_db.list_ingested_files()
        elif os.path.exists("./chroma_db"):
             # Temp init to get list if controller not ready (edge case)
             temp_db = VectorDatabase()
             return temp_db.list_ingested_files()
        return []

    current_files = get_file_list()
    
    if current_files:
        for file in current_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.caption(f"📄 {os.path.basename(file)}")
            with col2:
                if st.button("🗑️", key=f"del_{file}", help=f"Delete {file}"):
                    if st.session_state.rag_controller:
                        success = st.session_state.rag_controller.vector_db.delete_file(file)
                        if success:
                            st.success(f"Deleted {file}")
                            st.rerun()
                        else:
                            st.error("Deletion Failed")
    else:
        st.info("No documents found.")

    st.markdown("---")
    st.markdown("### ➕ Add New Document")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Drag and drop your PDF file here.")
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        # Check if this file is already processed in this session OR in DB
        # Note: We use the full path in DB, but filename check is a good first step
        # Ideally we check against the list we just fetched.
        if not any(file_name in f for f in current_files) and st.session_state.processed_file != file_name:
             with st.spinner(f"🚀 Ingesting & Chunking **{file_name}**..."):
                try:
                    # Save uploaded file to temporary path
                    save_dir = "data/temp"
                    os.makedirs(save_dir, exist_ok=True)
                    file_path = os.path.join(save_dir, file_name)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # --- BACKEND PIPELINE ---
                    # 1. Ingestion
                    ingestion = IngestionEngine()
                    doc_object, _, image_map = ingestion.process_file(file_path)
                    
                    # 2. Chunking
                    chunking = ChunkingEngine()
                    chunks = chunking.chunk_document(doc_object, image_map)
                    
                    # 3. Embedding & Storage
                    vector_db = VectorDatabase()
                    vector_db.create_or_update_vector_store(chunks)
                    
                    # 4. Initialize Controller (if not already)
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    
                    if st.session_state.rag_controller is None:
                         st.session_state.rag_controller = RAGController(vector_db, groq_api_key)
                    
                    # Refresh the controller's DB view just in case
                    st.session_state.rag_controller.vector_db = vector_db
                    
                    st.session_state.processed_file = file_name
                    
                    st.success(f"✅ added to Knowledge Base!")
                    st.rerun() # Rerun to update the file list
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    logging.error(f"Upload processing error: {e}")
        elif any(file_name in f for f in current_files):
             st.warning(f"File '{file_name}' already exists in the database.")
    
    st.markdown("---")
    st.markdown("### 🤖 System Status")
    if st.session_state.rag_controller:
        st.success("System Ready")
    else:
        st.warning("Waiting for Knowledge Base...")

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
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if st.session_state.rag_controller:
        with st.chat_message("assistant"):
            try:
                # Use Streaming with Self-Correction (CRAG)
                # stream_events is a generator yielding statuses and tokens
                stream_events = st.session_state.rag_controller.query_corrective_rag(prompt)
                
                # Placeholder for the final response
                response_placeholder = st.empty()
                full_response = ""
                context_docs = []

                # Create a status container for the reasoning steps
                with st.status("🧠 Thinking & Verifying...", expanded=True) as status:
                    for event in stream_events:
                        if event["type"] == "status":
                            # Update status container
                            status.write(event["content"])
                        elif event["type"] == "context":
                            # Capture context docs for image retrieval later
                            context_docs = event["content"]
                        elif event["type"] == "token":
                            # Append token to response
                            full_response += event["content"]
                            response_placeholder.markdown(full_response + "▌")
                    
                    status.update(label="✅ Answer Generated", state="complete", expanded=False)
                
                # Final render without cursor
                response_placeholder.markdown(full_response)
                
                # Save response to history
                response = full_response
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # --- Image Retrieval Logic ---
                # Check if we have extracted images for the documents in context
                found_images = set()
                
                # 1. Iterate context docs and check 'images' metadata
                for doc in context_docs:
                    if "images" in doc.metadata and doc.metadata["images"]:
                        # Metadata is a comma-separated string
                        imgs = doc.metadata["images"].split(",")
                        for img in imgs:
                            if img.strip():
                                found_images.add(img.strip())

                if found_images:
                    with st.expander("🖼️ Relevant Images / Figures", expanded=True):
                        # Display as Grid of Thumbnails
                        # We use a flexible grid
                        cols = st.columns(4) # 4 images per row
                        for idx, img_path in enumerate(sorted(list(found_images))):
                            with cols[idx % 4]:
                                # Check if file exists to be safe
                                if os.path.exists(img_path):
                                    st.image(
                                        img_path, 
                                        caption=f"Figure", 
                                        width=150, # Thumbnail size
                                        use_container_width=False 
                                        # Note: clicking expands naturally in Streamlit
                                    )
                                else:
                                    st.caption(f"Image not found: {os.path.basename(img_path)}")
                            
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        with st.chat_message("assistant"):
            st.warning("Please upload a document in the sidebar to start chatting.")
            
