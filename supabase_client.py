import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger("SupabaseClient")

url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_ANON_KEY", "")

# Initialize client if env vars are present
supabase: Optional[Client] = None
if url and key:
    try:
        supabase = create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

def create_chat_session(title: str = "New Chat") -> Optional[str]:
    """Creates a new chat session and returns its ID."""
    if not supabase: return None
    try:
        response = (
            supabase.table("chat_sessions")
            .insert({"title": title})
            .execute()
        )
        data = response.data
        if data and len(data) > 0:
            return data[0]["id"]
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
    return None

def get_all_chat_sessions() -> List[Dict[str, Any]]:
    """Fetches all past chat sessions ordered by newest first."""
    if not supabase: return []
    try:
        response = (
            supabase.table("chat_sessions")
            .select("id, title, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {e}")
        return []

def delete_chat_session(session_id: str) -> bool:
    """Deletes a chat session (messages are cascade deleted via SQL)."""
    if not supabase: return False
    try:
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        return False

def get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """Fetches all messages for a specific session ordered by creation time."""
    if not supabase: return []
    try:
        response = (
            supabase.table("chat_messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return []

def save_message(session_id: str, role: str, content: str) -> bool:
    """Saves a single message to a session."""
    if not supabase: return False
    try:
        supabase.table("chat_messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return False
