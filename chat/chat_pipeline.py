# Updated chat_pipeline.py
"""Enhanced Chat Pipeline with Persistent Memory and Multi-Session Support"""

import pandas as pd
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import text
import json
from datetime import datetime
import uuid

from config import OPENAI_API_KEY
from .langchain_chat_workflow import PostgresChatWorkflow

class ChatPipeline:
    """Enhanced chat pipeline with persistent memory and multi-session support"""
    
    def __init__(self, db_engine, content_generator=None, artist_list_manager=None):
        self.db_engine = db_engine
        self.content_generator = content_generator
        self.artist_list_manager = artist_list_manager
        self.workflow = PostgresChatWorkflow(db_engine)
        
        # Ensure chat memory table exists BEFORE loading history
        self._ensure_chat_memory_table()
    
    def _ensure_chat_memory_table(self):
        """Ensure chat_memory table exists with session management"""
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_memory (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        session_name VARCHAR(255),
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create index for better performance
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_chat_memory_session 
                        ON chat_memory(session_id, timestamp)
                    """))
                except Exception as index_error:
                    print(f"Note: Index might already exist: {index_error}")
                    
        except Exception as e:
            print(f"Error ensuring chat_memory table: {e}")
    
    def create_new_session(self, session_name: str = None) -> str:
        """Create a new chat session and return session_id"""
        session_id = str(uuid.uuid4())
        
        if not session_name:
            session_name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Initialize with system message
        # system_prompt = self.workflow._get_system_prompt()
        # self._save_message(session_id, session_name, "system", system_prompt)
        
        return session_id
    
    def _save_message(self, session_id: str, session_name: str, role: str, content: str):
        """Save a single message to database with session info"""
        try:
            with self.db_engine.begin() as conn:
                # For existing sessions, update session name if provided
                if session_name:
                    conn.execute(text("""
                        UPDATE chat_memory 
                        SET session_name = :session_name 
                        WHERE session_id = :session_id 
                        AND session_name IS NULL
                    """), {
                        "session_id": session_id,
                        "session_name": session_name
                    })
                
                # Save the message
                conn.execute(text("""
                    INSERT INTO chat_memory (session_id, session_name, role, content, timestamp)
                    VALUES (:session_id, :session_name, :role, :content, :timestamp)
                """), {
                    "session_id": session_id,
                    "session_name": session_name,
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now()
                })
        except Exception as e:
            print(f"Error saving message to database: {e}")
    
    def load_conversation_history(self, session_id: str) -> List:
        """Load conversation history for a specific session"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT role, content, timestamp 
                    FROM chat_memory 
                    WHERE session_id = :session_id 
                    ORDER BY timestamp ASC
                """), {"session_id": session_id})
                
                messages = []
                for row in result:
                    role, content, timestamp = row
                    if role == "system":
                        messages.append(SystemMessage(content=content))
                    elif role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                
                print(f"✅ Loaded {len(messages)} messages for session {session_id}")
                return messages
                
        except Exception as e:
            print(f"Note: Could not load conversation history: {e}")
            return []
    
    def ask_question(self, session_id: str, query: str, session_name: str = None) -> Dict[str, Any]:
        """Main method to handle user questions with session-specific memory"""
        # Load conversation history for this session
        conversation_history = self.load_conversation_history(session_id)
        
        # If no history exists, start with system prompt
        if not conversation_history:
            system_prompt = self.workflow._get_system_prompt()
            conversation_history = [SystemMessage(content=system_prompt)]
            # self._save_message(session_id, session_name, "system", system_prompt)
        
        # Add user message to conversation history and save to database
        conversation_history.append(HumanMessage(content=query))
        self._save_message(session_id, session_name, "user", query)
        
        try:
            # Limit conversation history to prevent context overflow
            if len(conversation_history) > 20:
                # Keep system message + last 19 messages
                system_msg = conversation_history[0]
                recent_msgs = conversation_history[-19:]
                conversation_history = [system_msg] + recent_msgs
            
            print(f"system message: {conversation_history[0].content}")
            # Execute LangChain workflow
            result = self.workflow.run(query, conversation_history)
            
            # Add assistant response to history and save to database
            conversation_history.append(AIMessage(content=result["answer"]))
            self._save_message(session_id, session_name, "assistant", result["answer"])
            
            return {
                "answer": result["answer"],
                "is_analytics": result.get("is_analytics", False),
                "used_rag": True,
                "sql_query": result.get("sql_query", ""),
                "query_result": result.get("query_result", pd.DataFrame())
            }
            
        except Exception as e:
            error_response = f"😅 I encountered an error processing your query: {str(e)}"
            
            # Save error response to database
            conversation_history.append(AIMessage(content=error_response))
            self._save_message(session_id, session_name, "assistant", error_response)
            
            return {
                "answer": error_response,
                "is_analytics": False,
                "used_rag": False
            }
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all chat sessions with metadata"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        session_id,
                        COALESCE(session_name, 'Unnamed Session') as session_name,
                        MIN(timestamp) as started_at,
                        MAX(timestamp) as last_activity,
                        COUNT(*) as message_count
                    FROM chat_memory 
                    GROUP BY session_id, session_name
                    ORDER BY last_activity DESC
                """))
                
                sessions = []
                for row in result:
                    sessions.append({
                        "session_id": row.session_id,
                        "session_name": row.session_name,
                        "started_at": row.started_at,
                        "last_activity": row.last_activity,
                        "message_count": row.message_count
                    })
                
                return sessions
                
        except Exception as e:
            print(f"Error getting sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text("""
                    DELETE FROM chat_memory WHERE session_id = :session_id
                """), {"session_id": session_id})
            
            print(f"✅ Chat session {session_id} deleted")
            return True
            
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a chat session"""
        try:
            with self.db_engine.begin() as conn:
                conn.execute(text("""
                    UPDATE chat_memory 
                    SET session_name = :new_name 
                    WHERE session_id = :session_id
                """), {
                    "session_id": session_id,
                    "new_name": new_name
                })
            
            print(f"✅ Session {session_id} renamed to {new_name}")
            return True
            
        except Exception as e:
            print(f"Error renaming session: {e}")
            return False
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a specific session"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_messages,
                        MIN(timestamp) as started_at,
                        MAX(timestamp) as last_activity,
                        COALESCE(session_name, 'Unnamed Session') as session_name
                    FROM chat_memory 
                    WHERE session_id = :session_id
                """), {"session_id": session_id})
                
                row = result.fetchone()
                if row:
                    return {
                        "session_name": row.session_name,
                        "total_messages": row.total_messages,
                        "started_at": row.started_at,
                        "last_activity": row.last_activity
                    }
                return {"total_messages": 0, "session_name": "Unknown Session"}
                
        except Exception as e:
            print(f"Error getting session summary: {e}")
            return {"total_messages": 0}
    
    def is_available(self) -> bool:
        """Check if pipeline is available"""
        return self.db_engine is not None
    
    def update_analytics_data(self, analytics_data, insights):
        """Update references - kept for compatibility"""
        pass