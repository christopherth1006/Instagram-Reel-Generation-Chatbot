import streamlit as st
from datetime import datetime
import time
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class ChatInterface:
    """Enhanced chat interface with multi-session support"""

    def __init__(self, chat_pipeline):
        self.chat_pipeline = chat_pipeline
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize chat session state with session management"""
        if "chat_initialized" not in st.session_state:
            st.session_state.chat_initialized = True
            st.session_state.show_memory_controls = False
            
        # Initialize current session
        if "current_session_id" not in st.session_state:
            # Create first session
            new_session_id = self.chat_pipeline.create_new_session(f"Chat Session {datetime.now().strftime('%H:%M')}")
            st.session_state.current_session_id = new_session_id
            st.session_state.current_session_name = f"Chat Session {datetime.now().strftime('%H:%M')}"
            
        # Load messages for current session
        if "messages" not in st.session_state:
            self._load_messages_for_current_session()

    def _load_messages_for_current_session(self):
        """Load messages for the current session"""
        session_id = st.session_state.current_session_id
        conversation_history = self.chat_pipeline.load_conversation_history(session_id)
        
        # Convert to chat format
        st.session_state.messages = []
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                st.session_state.messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage) and len(st.session_state.messages) == 0:
                # Use system message as first assistant message if no history
                st.session_state.messages.append({"role": "assistant", "content": msg.content})
        
        # If no messages, start with welcome
        if not st.session_state.messages:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hey there! 👋 I'm Cam, your music analytics buddy. I'm here to help you understand your Instagram performance and create awesome content. What would you like to explore today?",
                }
            ]

    def render_chat_interface(self):
        """Render the chat UI with session management"""
        st.title("💬 Chat with Cam - Your Music Analytics Assistant")

        # Session management in sidebar
        self._render_session_controls()

        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if user_input := st.chat_input("Ask me anything about your analytics or content..."):
            self._handle_user_input(user_input)

    def _render_session_controls(self):
        """Display session management controls in sidebar"""
        with st.sidebar:
            st.subheader("💭 Chat Sessions")
            
            # Get all sessions
            all_sessions = self.chat_pipeline.get_all_sessions()
            current_session_id = st.session_state.current_session_id
            
            # Create new session button
            if st.button("➕ New Session", use_container_width=True):
                new_session_name = f"Chat Session {datetime.now().strftime('%H:%M')}"
                new_session_id = self.chat_pipeline.create_new_session(new_session_name)
                st.session_state.current_session_id = new_session_id
                st.session_state.current_session_name = new_session_name
                self._load_messages_for_current_session()
                st.rerun()
            
            st.markdown("---")
            
            # Current session info
            if all_sessions:
                current_session = next((s for s in all_sessions if s["session_id"] == current_session_id), None)
                if current_session:
                    st.info(f"""
                    **Current Session:** {current_session['session_name']}
                    - Messages: {current_session['message_count']}
                    - Started: {current_session['started_at'].strftime('%m/%d %H:%M')}
                    """)
            
            # Session list
            if all_sessions:
                st.subheader("Your Sessions")
                
                for session in all_sessions:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        session_active = session["session_id"] == current_session_id
                        emoji = "🔵" if session_active else "⚪"
                        if st.button(
                            f"{emoji} {session['session_name']}", 
                            key=f"session_{session['session_id']}",
                            use_container_width=True,
                            help=f"Switch to {session['session_name']}"
                        ):
                            if not session_active:
                                st.session_state.current_session_id = session["session_id"]
                                st.session_state.current_session_name = session["session_name"]
                                self._load_messages_for_current_session()
                                st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete session"):
                            if len(all_sessions) > 1:
                                self.chat_pipeline.delete_session(session["session_id"])
                                if session["session_id"] == current_session_id:
                                    # Switch to another session
                                    other_sessions = [s for s in all_sessions if s["session_id"] != current_session_id]
                                    if other_sessions:
                                        st.session_state.current_session_id = other_sessions[0]["session_id"]
                                        st.session_state.current_session_name = other_sessions[0]["session_name"]
                                st.rerun()
                            else:
                                st.error("Cannot delete the only session")
                
                # Session management options
                with st.expander("Session Options"):
                    # Rename current session
                    new_name = st.text_input(
                        "Rename Current Session", 
                        value=st.session_state.get('current_session_name', ''),
                        key="rename_session_input"
                    )
                    if st.button("Rename Session", use_container_width=True) and new_name:
                        if self.chat_pipeline.rename_session(current_session_id, new_name):
                            st.session_state.current_session_name = new_name
                            st.success("Session renamed!")
                            st.rerun()
                    
                    # Clear current session
                    if st.button("🧹 Clear Current Session", use_container_width=True):
                        self._clear_current_session()
                        
            else:
                st.info("No chat sessions yet. Start a conversation!")

    def _handle_user_input(self, user_input: str):
        """Process user query and generate response with session memory"""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = self.chat_pipeline.ask_question(
                        st.session_state.current_session_id,
                        user_input,
                        st.session_state.current_session_name
                    )
                    ai_response = response.get("answer", "I'm not sure how to answer that right now.")
                    
                    # Display with typing effect
                    message_placeholder = st.empty()
                    self._display_with_typing_effect(message_placeholder, ai_response)

                    # Update session state messages (already saved to DB via pipeline)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

                except Exception as e:
                    error_msg = f"Oops! I ran into an issue: {str(e)}. Could you try asking that again?"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    def _display_with_typing_effect(self, placeholder, text, speed=0.01):
        """Simulate typing effect for better UX"""
        # For shorter responses, show immediately
        if len(text) < 100:
            placeholder.markdown(text)
            return
            
        # For longer responses, simulate typing
        current_text = ""
        for char in text:
            current_text += char
            placeholder.markdown(current_text + "▌")
            time.sleep(speed)
        placeholder.markdown(current_text)

    def _clear_current_session(self):
        """Clear current session conversation"""
        session_id = st.session_state.current_session_id
        self.chat_pipeline.delete_session(session_id)
        
        # Create new empty session with same name
        session_name = st.session_state.current_session_name
        new_session_id = self.chat_pipeline.create_new_session(session_name)
        st.session_state.current_session_id = new_session_id
        
        # Reset messages
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! 🎉 I'm ready for a fresh conversation. What would you like to know about your analytics or what kind of content should we work on?",
            }
        ]
        
        st.success("Current session cleared!")
        st.rerun()