"""
Content Generator - Main Orchestrator - Updated to use PostgreSQL
Coordinates content generation across different types (reels, quizzes, polls)
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
import pandas as pd
from config import OPENAI_MODEL, OPENAI_TEMPERATURE, BRAND_VOICE, REEL_BATCH_SIZE, QUIZ_BATCH_SIZE, POLL_BATCH_SIZE

from .scoring_engines import QuizScoringEngine, ReelScoringEngine
from .quiz_generator import QuizGenerator
from .poll_generator import PollGenerator
from .reel_generator import ReelGenerator

class ContentGenerator:
    """Main content generator that orchestrates all content types using PostgreSQL"""
    
    def __init__(self, api_key: str, artist_list_manager=None, db_engine=None):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=api_key
        )
        self.brand_voice = BRAND_VOICE
        self.artist_list_manager = artist_list_manager
        self.db_engine = db_engine
        
        # Initialize scoring engines
        self.quiz_scoring_engine = QuizScoringEngine()
        self.reel_scoring_engine = ReelScoringEngine()
        
        # Initialize content generators with postgres manager
        self.quiz_generator = QuizGenerator(self.llm, self.brand_voice, artist_list_manager)
        self.poll_generator = PollGenerator(self.llm, self.brand_voice, artist_list_manager)
        self.reel_generator = ReelGenerator(self.llm, self.brand_voice, artist_list_manager, db_engine)
    
    def generate_reel_scripts(self, 
                            insights: Dict, 
                            batch_size: int = REEL_BATCH_SIZE,
                            focus_themes: Optional[List[str]] = None,
                            analytics_data: Optional[pd.DataFrame] = None,
                            chat_history: List = []
                            ) -> pd.DataFrame:
        """Generate Reel scripts using PostgreSQL data"""
        return self.reel_generator.generate_reel_scripts(insights, batch_size, focus_themes, analytics_data, chat_history)
    
    def generate_story_quizzes(self, 
                             insights: Dict,
                             batch_size: int = QUIZ_BATCH_SIZE,
                             quiz_types: Optional[List[str]] = None,
                             difficulty: str = "medium",
                             chat_history: List = []
                             ) -> pd.DataFrame:
        """Generate Story quizzes using PostgreSQL data"""
        return self.quiz_generator.generate_story_quizzes(insights, batch_size, quiz_types, difficulty, chat_history)
    
    def generate_story_polls(self, insights: Dict = None, batch_size: int = 5, 
                        user_query: Optional[str] = None, difficulty: str = "medium",
                        num_options: int = 2, chat_history: List = None) -> pd.DataFrame:
        """Generate story polls with specified number of options"""
        if not self.poll_generator:
            raise Exception("Poll generator not initialized")
        
        return self.poll_generator.generate_story_polls(
            insights=insights,
            batch_size=batch_size,
            user_query=user_query,
            difficulty=difficulty,
            num_options=num_options,
            chat_history=chat_history or []
        )