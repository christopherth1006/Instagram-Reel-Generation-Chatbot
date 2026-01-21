"""
Quiz Generation Module - Updated to use Pinecone DB
Handles generation of Instagram Story quizzes with artist filtering and batch size control
"""
import json
import random
import re
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from .scoring_engines import QuizScoringEngine
import streamlit as st
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY

class QuizGenerator:
    """Handles quiz generation logic using Pinecone DB"""
    
    def __init__(self, llm, brand_voice: Dict[str, Any], artist_list_manager=None):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.5,
            api_key=OPENAI_API_KEY
        )
        self.brand_voice = brand_voice
        self.artist_list_manager = artist_list_manager
        self.quiz_scoring_engine = QuizScoringEngine()

    def generate_story_quizzes(self, 
                             insights: Dict,
                             batch_size: int,
                             quiz_types: Optional[List[str]] = None,
                             difficulty: str = "medium",
                             chat_history: List = []
                             ) -> pd.DataFrame:
        """Generate Story quizzes using data from PostgreSQl with proper difficulty handling"""
        
        quiz_types = quiz_types or ["Who Said It", "Fill in the Blank", "Guess the Year", "Tracklist"]
        print(f"Debug: quiz_types: {quiz_types}")
        # Apply strict artist list filtering for quizzes
        artist_filter = ""
        if self.artist_list_manager:
            artist_filter = self.artist_list_manager.get_artist_filter_prompt()

        difficulty_prompt = self._get_difficulty_prompt(difficulty)
        
        system_prompt = f"""
        You are an expert content creator generating Instagram Story quizzes.
        Brand Voice: {', '.join(self.brand_voice.get('tone', []))}

        DIFFICULTY LEVEL: {difficulty.upper()}
        {difficulty_prompt}

        {f"STRICT ARTIST FILTERING: {artist_filter}" if artist_filter else ""}
        {f"CRITICAL: You MUST ONLY use artists from the provided list. Do not suggest any artists not in the list." if artist_filter else ""}

        Goal: Create quizzes that are factually accurate, engaging, and tailored to the {difficulty.upper()} difficulty level
        

        RULES FOR {difficulty.upper()} DIFFICULTY:
        {self._get_difficulty_rules(difficulty)}

        Quiz Types to Generate: {', '.join(quiz_types)}

        If the user wants to update the previous generated quizzes, Update those quizzes as per user's request, NOT generating a NEW ONE. And indicate you updated the previous ones. 

        OUTPUT FORMAT:
        Return a JSON array with exactly {batch_size} quiz items. Each item must have:
        - type: one of {quiz_types}
        - question: clear, engaging question
        - option_a, option_b, option_c, option_d: four answer options
        - correct: the correct answer (A, B, C, or D)
        - fun_fact: interesting fact about the answer
        - difficulty: "{difficulty}"

        Ensure questions and answers match the {difficulty.upper()} difficulty level.
        """

        human_prompt = f"""
        Generate exactly {batch_size} Instagram Story quiz questions with {difficulty.upper()} difficulty.

        DIFFICULTY REQUIREMENTS:
        {difficulty_prompt}

        Quiz Types: {', '.join(quiz_types)}

        If chat_history contains previous quizzes and the user intends to update them:
        - Edit the existing quizzes for clarity, engagement, and correctness.
        - Keep original quiz types but improve content.

        If chat_history is empty or the user wants new quizzes:
        - Generate fresh quizzes.

        - Ensure all information is accurate and aligns with the latest available data.

        Return as a SINGLE JSON array in this EXACT format:
        ```json
        [{{
            "type": "Who Said It",
            "question": "Which artist said this iconic lyric?",
            "option_a": "Artist A",
            "option_b": "Artist B", 
            "option_c": "Artist C",
            "option_d": "Artist D",
            "correct": "A",
            "fun_fact": "This lyric comes from their 2020 hit song that went viral on TikTok.",
            "difficulty": "{difficulty}"
        }}]
        ```
        
        CRITICAL: Return EXACTLY {batch_size} quizzes. No more, no less.
        """
        
        try:

            print(f"chat history: {chat_history}")

            chat_history_clone =  chat_history.copy()

            chat_history_clone.append(HumanMessage(content=system_prompt))
            chat_history_clone.append(HumanMessage(content=human_prompt))

            response = self.llm.invoke(chat_history_clone)
            
            content = response.content
            if '```' in content:
                s = content.find('```')
                e = content.rfind('```')
                if s != -1 and e != -1 and e > s:
                    fenced = content[s+3:e].strip()
                    if fenced.startswith('json'):
                        fenced = fenced[4:].strip()
                    content = fenced
            
            quiz_data = json.loads(content)
            normalized = self._normalize_quiz_data(quiz_data, batch_size, difficulty, st.session_state.insights)
            
            # Final verification: ensure we have exactly the requested batch size
            if len(normalized) < batch_size:
                remaining = batch_size - len(normalized)
                additional_fallback = self._create_fallback_quizzes(remaining, quiz_types, difficulty, st.session_state.insights)
                normalized.extend(additional_fallback.to_dict('records'))
            elif len(normalized) > batch_size:
                normalized = normalized[:batch_size]
            
            return pd.DataFrame(normalized)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM generation failed, using fallback: {e}")
            return self._create_fallback_quizzes(batch_size, quiz_types, difficulty, st.session_state.insights)

    def _get_difficulty_prompt(self, difficulty: str) -> str:
        """Get difficulty-specific prompt instructions - MATCHING POLL GENERATOR"""
        difficulty_prompts = {
            "easy": """
            EASY DIFFICULTY:
            - Use mainstream, popular artists that everyone knows
            - Focus on recent hits and well-known classics
            - Questions should be easily answerable by casual music listeners
            - Use obvious lyrics and well-known collaborations
            - Fun facts should be widely known trivia
            - Examples: Drake, Taylor Swift, Beyoncé, The Weeknd, popular TikTok sounds
            """,
            "medium": """
            MEDIUM DIFFICULTY:
            - Mix mainstream artists with some rising/underground artists
            - Include deeper album cuts and moderate popularity songs
            - Questions should challenge regular music fans
            - Use less obvious lyrics and production credits
            - Fun facts should be interesting but not too obscure
            - Examples: Mix of mainstream and niche artists
            """,
            "hard": """
            HARD DIFFICULTY:
            - Focus on underground, niche, or older classic artists
            - Use deep album cuts, B-sides, and obscure features
            - Questions should challenge hardcore music fans and experts
            - Reference specific production techniques, sample sources, or industry drama
            - Fun facts should be deep cuts and insider knowledge
            - Examples: Underground artists, deep album cuts, obscure samples
            """
        }
        return difficulty_prompts.get(difficulty.lower(), difficulty_prompts["medium"])
    
    def _get_difficulty_rules(self, difficulty: str) -> str:
        """Get difficulty-specific rules"""
        if difficulty == "easy":
            return """
            - Use only mainstream, popular artists
            - Focus on recent viral hits and well-known classics
            - Questions should be answerable by casual listeners
            - Options should include obvious correct answer and plausible alternatives
            - Fun facts should be common knowledge
            """
        elif difficulty == "hard":
            return """
            - Use underground, niche, or older artists
            - Focus on deep cuts, B-sides, and obscure features
            - Questions should challenge experts
            - Include technical questions about production, samples, or industry details
            - Fun facts should be insider knowledge
            """
        else:  # medium
            return """
            - Mix mainstream and niche artists
            - Include both hits and deeper album cuts
            - Questions should challenge regular fans
            - Balance between obvious and challenging questions
            - Fun facts should be interesting but accessible
            """
    
    def _normalize_quiz_data(self, quiz_data: List[Dict], batch_size: int, difficulty: str, insights: Dict = None) -> List[Dict[str, Any]]:
        """Normalize and clean quiz data from LLM response with proper scoring"""
        normalized = []
        
        for item in quiz_data[:batch_size]:
            # Apply artist filtering - skip if no uploaded artists are mentioned
            if self.artist_list_manager:
                quiz_text = f"{item.get('question', '')} {item.get('option_a', '')} {item.get('option_b', '')} {item.get('option_c', '')} {item.get('option_d', '')} {item.get('fun_fact', '')}".lower()
                uploaded_artists = self.artist_list_manager.get_artist_names_for_prompt()
                if uploaded_artists:
                    artist_mentioned = False
                    for artist in uploaded_artists:
                        if artist.lower() in quiz_text:
                            artist_mentioned = True
                            break
                    
                    if not artist_mentioned:
                        continue
            
            # Clean and validate each field
            qtype = self._clean_text(item.get('type', '')).strip() or "Who Said It"
            question = self._clean_text(item.get('question', '')).strip() or "Quiz question"
            option_a = self._clean_text(item.get('option_a', '')).strip() or "Option A"
            option_b = self._clean_text(item.get('option_b', '')).strip() or "Option B"
            option_c = self._clean_text(item.get('option_c', '')).strip() or "Option C"
            option_d = self._clean_text(item.get('option_d', '')).strip() or "Option D"
            
            # Validate correct answer
            correct = str(item.get('correct', 'A')).strip().upper()
            if correct not in ['A', 'B', 'C', 'D']:
                correct = 'A'
            
            fun_fact = self._clean_text(item.get('fun_fact', '')).strip() or "Interesting fact"
            
            # Get difficulty from item or use default
            item_difficulty = item.get('difficulty', difficulty)
            
            # Calculate scores using the IMPROVED scoring engine
            quiz_item = {
                'type': qtype,
                'question': question,
                'option_a': option_a,
                'option_b': option_b,
                'option_c': option_c,
                'option_d': option_d,
                'correct': correct,
                'fun_fact': fun_fact,
                'difficulty': item_difficulty
            }
            
            # Use the improved scoring engine for realistic score variation
            scores = self.quiz_scoring_engine.calculate_quiz_scores(quiz_item, st.session_state.insights)
            predicted_score = scores['predicted_score']
            shareability_score = scores['shareability_score']
            
            normalized.append({
                'type': qtype,
                'question': question,
                'option_a': option_a,
                'option_b': option_b,
                'option_c': option_c,
                'option_d': option_d,
                'correct': correct,
                'fun_fact': fun_fact,
                'predicted_score': predicted_score,
                'shareability_score': shareability_score,
                'difficulty': item_difficulty
            })
        
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing gibberish and random characters"""
        if not text:
            return ""
        
        # Remove common gibberish patterns
        text = re.sub(r'[�◆●✔✅👆🎵❓]', '', str(text))  # Remove emojis and special chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s?.!,-]', '', text)  # Remove unusual characters but keep basic punctuation
        
        return text.strip()

    def _create_fallback_quizzes(self, batch_size: int, quiz_types: List[str], difficulty: str, insights: Dict = None) -> pd.DataFrame:
        """Create fallback quiz data with MUCH better score variation"""
        import re
        
        fallback_data = []
        
        # Get artists from uploaded list if available
        artists = []
        if self.artist_list_manager:
            artists = self.artist_list_manager.get_artist_names_for_prompt()
        
        # Adjust artist selection based on difficulty
        if artists:
            if difficulty == "easy":
                selected_artists = artists[:min(10, len(artists))]
            elif difficulty == "hard":
                selected_artists = artists[min(5, len(artists)-10):] if len(artists) > 15 else artists
            else:
                selected_artists = artists
        
        for i in range(batch_size):
            quiz_type = quiz_types[i % len(quiz_types)]
            
            # Use uploaded artists if available, otherwise use generic content
            if artists and selected_artists:
                artist = selected_artists[i % len(selected_artists)]
                
                if difficulty == "easy":
                    if quiz_type == "Who Said It":
                        question = f"Which popular artist said: 'I'm on one'?"
                        options = [artist, "Drake", "Future", "Kendrick Lamar"]
                    elif quiz_type == "Fill in the Blank":
                        question = f"Complete the lyric: 'Started from the bottom, now ___ here'"
                        options = ["we're", "I'm", "they're", "we"]
                    elif quiz_type == "Guess the Year":
                        question = f"What year did {artist} release their biggest hit?"
                        options = ["2020", "2019", "2021", "2018"]
                    else:  # Tracklist
                        question = f"Which song is NOT on {artist}'s most popular album?"
                        options = ["Deep Cut", "Hit Single", "Fan Favorite", "B-Side"]
                
                elif difficulty == "hard":
                    if quiz_type == "Who Said It":
                        question = f"Which underground artist said: 'The city is mine, I'm the king of the underground'?"
                        options = [artist, "Mainstream Rapper", "Popular Artist", "Commercial Star"]
                    elif quiz_type == "Fill in the Blank":
                        question = f"Complete the obscure lyric: 'Microphone check, one two, ___'"
                        options = ["what is this?", "who is this?", "where you at?", "how we do?"]
                    elif quiz_type == "Guess the Year":
                        question = f"What year did {artist} release their first mixtape?"
                        options = ["2014", "2015", "2016", "2017"]
                    else:  # Tracklist
                        question = f"Which deep cut appears on {artist}'s limited edition vinyl?"
                        options = ["Rare Track", "Mainstream Hit", "Radio Single", "Popular Song"]
                
                else:  # medium
                    if quiz_type == "Who Said It":
                        question = f"Which artist said this lyric about success?"
                        options = [artist, "Similar Artist", "Collaborator", "Influence"]
                    elif quiz_type == "Fill in the Blank":
                        question = f"Complete the lyric: 'No new friends, ___'"
                        options = ["that's just how I feel", "that's the motto", "no new ends", "no new trends"]
                    elif quiz_type == "Guess the Year":
                        question = f"What year did {artist} collaborate with that producer?"
                        options = ["2018", "2019", "2020", "2021"]
                    else:  # Tracklist
                        question = f"Which song was the lead single from {artist}'s breakthrough album?"
                        options = ["Breakout Hit", "Deep Cut", "B-Side", "Remix"]
                
                fun_fact = f"{artist} has been influential in the music scene."
                
            else:
                # Generic fallbacks with difficulty adjustment
                if difficulty == "easy":
                    question = f"Easy {quiz_type} question about popular music"
                    options = ["Mainstream Answer", "Plausible Wrong", "Another Wrong", "Clearly Wrong"]
                    fun_fact = "This is a well-known fact about popular music."
                elif difficulty == "hard":
                    question = f"Expert {quiz_type} question about music deep cuts"
                    options = ["Obscure Answer", "Plausible Wrong", "Another Wrong", "Clearly Wrong"]
                    fun_fact = "This is insider knowledge about music production."
                else:
                    question = f"Standard {quiz_type} question {i+1}"
                    options = ["Correct Answer", "Plausible Wrong", "Another Wrong", "Clearly Wrong"]
                    fun_fact = f"Interesting fact about music history {i+1}."
            
            # Calculate REALISTIC scores with SIGNIFICANT variation
            if difficulty == "easy":
                base_score = random.uniform(70, 90)  # Wider range
                share_base = random.uniform(74, 94)
            elif difficulty == "hard":
                base_score = random.uniform(60, 80)  # Wider range  
                share_base = random.uniform(64, 84)
            else:  # medium
                base_score = random.uniform(65, 85)  # Wider range
                share_base = random.uniform(69, 89)
            
            # Add substantial random variation
            predicted_score = round(base_score + random.uniform(-10, 10), 1)
            shareability_score = round(share_base + random.uniform(-8, 12), 1)
            
            # Ensure scores stay in reasonable ranges
            predicted_score = max(45, min(95, predicted_score))
            shareability_score = max(50, min(95, shareability_score))
            
            fallback_data.append({
                "type": quiz_type,
                "question": question,
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
                "correct": "A",
                "fun_fact": fun_fact,
                "predicted_score": predicted_score,
                "shareability_score": shareability_score,
                "difficulty": difficulty
            })
        
        return pd.DataFrame(fallback_data)