"""
Poll Generation Module - Updated to use PostgreSQL
Handles generation of Instagram Story polls with artist filtering and batch size control
"""
import json
import random
import re
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from sqlalchemy import text
import streamlit as st

class PollGenerator:
    """Handles poll generation logic using PostgreSQL"""
    
    def __init__(self, llm, brand_voice: Dict[str, Any], artist_list_manager=None, db_engine=None):
        self.llm = llm
        self.brand_voice = brand_voice
        self.artist_list_manager = artist_list_manager
        self.db_engine = db_engine
        self.last_extracted_examples = []  # Store examples extracted from user query

    def generate_story_polls(self, 
                           insights: Dict = None,
                           batch_size: int = 5,
                           user_query: Optional[str] = None,
                           difficulty: str = "medium",
                           num_options: int = 2,  # New parameter: 2 or 4 options
                           return_extracted_examples: bool = False,
                           chat_history: List = []
                           ) -> pd.DataFrame:
        """Generate balanced Story polls using data from PostgreSQL with improved few-shot learning and difficulty levels"""
        
        context = self._build_context_from_postgres()
        
        # Apply strict artist list filtering for polls
        artist_filter = ""
        if self.artist_list_manager:
            artist_filter = self.artist_list_manager.get_artist_filter_prompt()
            if artist_filter:
                context += f"\n\nSTRICT ARTIST FILTERING: {artist_filter}"
                context += "\n\nCRITICAL: You MUST ONLY use artists from the provided list. Do not suggest any artists not in the list."
        
        # Also extract examples from user query if provided (as additional context)
        if user_query:
            query_examples = self._extract_clean_example_polls_from_query(user_query)
            if query_examples:
                self.last_extracted_examples = query_examples  # Store for later saving
            else:
                self.last_extracted_examples = []
        else:
            self.last_extracted_examples = []
        
        # Add difficulty-specific instructions
        difficulty_prompt = self._get_difficulty_prompt(difficulty)
        
        # Add option-specific instructions
        option_prompt = self._get_option_prompt(num_options)
        
        system_prompt = f"""
        You are creating Instagram Story polls for {self.brand_voice.get('persona_name', 'a music curator')}.
        
        Brand Voice: {', '.join(self.brand_voice.get('tone', []))}
        
        DIFFICULTY LEVEL: {difficulty.upper()}
        {difficulty_prompt}

        POLL TYPE: {num_options}-OPTION POLLS
        {option_prompt}

        Performance Context from User's Analytics:
        {context}

        Goal: Create CLEAR, ENGAGING polls designed for balanced engagement.

        RULES:
        - Make poll questions CLEAR and UNDERSTANDABLE
        - Use proper English grammar and spelling
        - Options should be realistic and comparable
        - Balance scores should be between 85-98 (representing how balanced the poll is)
        - Predicted splits should be realistic
        - Themes should be relevant to music/artists
        - Do NOT include random characters or gibberish

        CREATIVITY & VARIETY REQUIREMENTS:

        1. ARTIST DIVERSITY:
           - Use DIFFERENT artists across all {batch_size} polls
           - Maximum 1 appearance per artist in the entire batch
           - Include mix of genres (rap, R&B, pop, rock, electronic)
           - Include both mainstream and niche artists (adjusted for difficulty)
           - Mix current stars with classic artists

        2. QUESTION VARIETY:
           Alternate between these question styles across the batch:
           - "Whose [quality] fits your [context] more?"
           - "Which [element] defines [situation] better?" 
           - "Who would you rather [action] with?"
           - "Which [aspect] resonates more with you?"
           - "What's your preferred [scenario] soundtrack?"
           - "Which artist's [specific trait] hits harder?"
           - "Whose music matches [mood/occasion] better?"

        3. CREATIVE ANGLES (adjusted for difficulty):
           {self._get_difficulty_specific_angles(difficulty)}

        4. POLL CATEGORIES (rotate through these):
           - Artist vs Artist comparisons
           - Song vs Song battles  
           - Era preferences (90s vs 2000s vs 2010s)
           - Genre debates (Rap vs R&B, Pop vs Rock)
           - Album showdowns
           - Producer battles
           - Feature artist face-offs
           - Music video aesthetics

        DO NOT REPEAT:
        - The same artist more than once
        - Simple "Who's better?" phrasing
        - "Who has better flow?" (overused)
        - Generic A vs. B without creative context
        - The same question structure back-to-back

        If the user wants to update the previous generated polls, Update those polls as per user's request, NOT generating a NEW ONE. And indicate you updated the previous ones. 

        ARTIST POOL SUGGESTIONS: {self._get_difficulty_artist_suggestions(difficulty)}
        """
        
        # Add few-shot examples if provided
        if self.last_extracted_examples:
            system_prompt += f"\n\n{'='*80}\n"
            system_prompt += f"FEW-SHOT LEARNING EXAMPLES ({len(self.last_extracted_examples)} examples provided):\n"
            system_prompt += f"{'='*80}\n"
            system_prompt += f"\nIMPORTANT: Study these examples carefully and generate new polls that match this quality and style.\n"
            system_prompt += f"These are examples of BALANCED, ENGAGING polls that perform well:\n\n"
            
            for i, example in enumerate(self.last_extracted_examples, 1):
                system_prompt += f"--- EXAMPLE {i} ---\n"
                system_prompt += f"Theme: {example.get('theme', '')}\n"
                system_prompt += f"Question: {example.get('prompt', '')}\n"
                system_prompt += f"Option A: {example.get('option_a', '')}\n"
                system_prompt += f"Option B: {example.get('option_b', '')}\n"
                if num_options == 4:
                    system_prompt += f"Option C: {example.get('option_c', '')}\n"
                    system_prompt += f"Option D: {example.get('option_d', '')}\n"
                system_prompt += f"Predicted Split: {example.get('predicted_split', '50% A / 50% B')}\n"
                system_prompt += f"Balance Score: {example.get('balance_score', 95.0)}\n"
                
                # Add notes if available
                if example.get('notes'):
                    system_prompt += f"Why it works: {example.get('notes', '')}\n"
                system_prompt += f"\n"
            
            system_prompt += f"{'='*80}\n"
            system_prompt += f"Use these examples as templates for quality, style, and balance.\n"
            system_prompt += f"Create NEW polls with similar characteristics but different content.\n"
            system_prompt += f"{'='*80}\n\n"
        
        # Format the JSON structure based on number of options
        if num_options == 2:
            json_format = """```json
            [{
                "theme": "Artist Vibe Match",
                "prompt": "Whose sound fits your morning routine better?",
                "option_a": "Chill Frank Ocean vibes",
                "option_b": "Energetic Megan Thee Stallion", 
                "predicted_split": "51% A / 49% B",
                "balance_score": 95.0,
                "best_balanced_option": "Option A",
                "alternates": "SZA; Doja Cat",
                "predicted_score": 84.2,
                "difficulty": "{difficulty}"
            }]```"""
        else:  # 4 options
            json_format = """```json
            [{
                "theme": "Artist Vibe Match",
                "prompt": "Which artist's vibe matches your current mood?",
                "option_a": "Chill Frank Ocean vibes",
                "option_b": "Energetic Megan Thee Stallion",
                "option_c": "Moody The Weeknd energy",
                "option_d": "Upbeat Doja Cat fun",
                "predicted_split": "25% A / 30% B / 25% C / 20% D",
                "balance_score": 90.0,
                "best_balanced_option": "Option B",
                "alternates": "SZA; Drake; Beyoncé",
                "predicted_score": 82.5,
                "difficulty": "{difficulty}"
            }]```"""
        
        human_prompt = f"""
        Generate EXACTLY {batch_size} CLEAN, PROFESSIONAL Instagram Story poll ideas.

        DIFFICULTY: {difficulty.upper()}
        {difficulty_prompt}

        POLL TYPE: {num_options}-OPTION POLLS
        {option_prompt}

        CRITICAL REQUIREMENTS:
        1. ARTIST DIVERSITY: Each artist can only appear ONCE in the entire batch of {batch_size} polls
        2. QUESTION VARIETY: Use different question phrasing for each poll - no repeated structures
        3. CREATIVE ANGLES: Explore different comparison types beyond simple "better" questions
        4. DIFFICULTY APPROPRIATE: Ensure questions match the {difficulty.upper()} difficulty level

        Each poll MUST have:
        1. A clear, relevant theme from the categories listed
        2. A well-worded, creative question that makes sense
        3. {num_options} realistic, comparable options with different artists
        4. A realistic predicted split that sums to 100%
        5. A balance score between 85-98
        6. Proper English grammar and spelling
        7. A difficulty level of "{difficulty}"

        If chat_history contains previous polls and the user intends to update them:
        - Edit the existing polls for clarity, engagement, and correctness.
        - Keep original poll types but improve content.

        QUESTION STYLE EXAMPLES (rotate through these):
        {self._get_difficulty_question_styles(difficulty)}

        {"IMPORTANT: Follow the creative style and diversity of the examples, ensuring no artist repetition." if self.last_extracted_examples else ""}

        Return as a SINGLE JSON array in this EXACT format:
        {json_format}

        ARTIST DIVERSITY ENFORCEMENT: 
        - Track artists used across all {batch_size} polls
        - If you run out of mainstream artists, include niche or international artists
        - Never use the same artist twice in this batch

        CRITICAL: Return EXACTLY {batch_size} polls. No more, no less.

        User Input:
        {user_query}
        """
        
        try:
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
            
            poll_data = json.loads(content)
            normalized = self._normalize_poll_data(poll_data, batch_size, difficulty, num_options)
            
            # Final verification: ensure we have exactly the requested batch size
            if len(normalized) < batch_size:
                remaining = batch_size - len(normalized)
                additional_fallback = self._create_clean_fallback_polls(remaining, difficulty, num_options)
                normalized.extend(additional_fallback.to_dict('records'))
            elif len(normalized) > batch_size:
                normalized = normalized[:batch_size]
            
            return pd.DataFrame(normalized)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM generation failed, using fallback: {e}")
            return self._create_clean_fallback_polls(batch_size, difficulty, num_options)

    def _get_option_prompt(self, num_options: int) -> str:
        """Get option-specific prompt instructions"""
        if num_options == 4:
            return """
            4-OPTION POLLS:
            - Create questions that work well with 4 distinct choices
            - Ensure all 4 options are equally compelling and balanced
            - Use question types that naturally have multiple good answers
            - Predicted splits should be more distributed (e.g., "25% A / 30% B / 25% C / 20% D")
            - Balance scores may be slightly lower but should still be above 85
            - Questions like "Which genre fits your mood?" or "Which era was the best?" work well
            """
        else:
            return """
            2-OPTION POLLS:
            - Create clear binary choices
            - Options should be direct competitors or clear alternatives
            - Aim for close splits (45-55% range) to maximize engagement
            - Balance scores should be high (90+)
            - Questions like "A vs B" or "This or That" work well
            """



    def _get_difficulty_prompt(self, difficulty: str) -> str:
        """Get difficulty-specific prompt instructions"""
        difficulty_prompts = {
            "easy": """
            EASY DIFFICULTY:
            - Use mainstream, popular artists that everyone knows
            - Focus on recent hits and well-known classics
            - Questions should be easily answerable by casual music listeners
            - Compare obvious attributes (voice, popularity, recent hits)
            - Use familiar comparisons that spark immediate opinions
            """,
            "medium": """
            MEDIUM DIFFICULTY:
            - Mix mainstream artists with some rising/underground artists
            - Include deeper album cuts and moderate popularity songs
            - Questions should challenge regular music fans
            - Compare specific styles, eras, or artistic approaches
            - Include some niche genres or specific production styles
            """,
            "hard": """
            HARD DIFFICULTY:
            - Focus on underground, niche, or older classic artists
            - Use deep album cuts, B-sides, and obscure features
            - Questions should challenge hardcore music fans and experts
            - Compare specific production techniques, lyrical themes, or cultural impact
            - Reference specific eras, regional scenes, or industry influences
            - Include international artists and specialized genres
            """
        }
        return difficulty_prompts.get(difficulty.lower(), difficulty_prompts["medium"])
    
    def _get_difficulty_specific_angles(self, difficulty: str) -> str:
        """Get difficulty-specific creative angles"""
        if difficulty == "easy":
            return """
            - Vibe/mood matching (chill vs. energetic)
            - Party vs. chill music preferences
            - Mainstream popularity comparisons
            - Recent hit vs. recent hit
            - Voice/style preferences
            """
        elif difficulty == "hard":
            return """
            - Specific production technique preferences
            - Lyrical depth vs. musical complexity
            - Cultural impact and legacy comparisons
            - Obscure sample sources or influences
            - Regional scene representations
            - Technical skill comparisons (flow, wordplay, delivery)
            """
        else:  # medium
            return """
            - Vibe/mood matching (chill vs. energetic)
            - Lyric depth vs. production quality  
            - Era-defining impact (which shaped their generation more)
            - Collaboration potential (unexpected pairings)
            - Live performance energy
            - Cultural influence and legacy
            """
    
    def _get_difficulty_artist_suggestions(self, difficulty: str) -> str:
        """Get difficulty-appropriate artist suggestions"""
        if difficulty == "easy":
            return "Draw from mainstream artists like Drake, Taylor Swift, Beyoncé, Bad Bunny, The Weeknd, Olivia Rodrigo, Harry Styles, Doja Cat, etc."
        elif difficulty == "hard":
            return "Draw from niche/underground artists like Little Simz, Smino, Noname, Moses Sumney, Blood Orange, Arca, FKA twigs, and classic influential artists from specific eras and regions."
        else:  # medium
            return "Draw from diverse artists like Drake, Kendrick Lamar, Beyoncé, Taylor Swift, Bad Bunny, SZA, Future, Travis Scott, J. Cole, Doja Cat, Lil Nas X, Olivia Rodrigo, The Weeknd, Megan Thee Stallion, Tyler The Creator, Billie Eilish, Rosalía, Harry Styles, and classic artists."
    
    def _get_difficulty_question_styles(self, difficulty: str) -> str:
        """Get difficulty-appropriate question styles"""
        if difficulty == "easy":
            return """
            - "Whose sound fits your current vibe better?"
            - "Which track defines summer for you?"
            - "Who would you rather see at a festival?"
            - "Which artist's new album are you more excited for?"
            - "What's your go-to party soundtrack?"
            """
        elif difficulty == "hard":
            return """
            - "Whose lyrical craftsmanship resonates deeper with you?"
            - "Which artist's production style is more innovative?"
            - "Who had more cultural impact on their specific genre?"
            - "Which deep cut deserves more recognition?"
            - "Whose artistic evolution is more impressive?"
            """
        else:  # medium
            return """
            - "Whose sound fits your morning routine better?"
            - "Which track defines summer nights for you?"
            - "Who would you rather see live in concert?"
            - "Which artist's lyrics hit deeper personally?"
            - "What's your go-to workout soundtrack vibe?"
            - "Whose new album are you more excited for?"
            - "Which collaboration would be more unexpected fire?"
            """

    def _normalize_poll_data(self, poll_data: List[Dict], batch_size: int, difficulty: str, num_options: int) -> List[Dict[str, Any]]:
        """Normalize and clean poll data from LLM response with difficulty and option count"""
        normalized = []
        
        for item in poll_data[:batch_size]:
            # Apply artist filtering - skip if no uploaded artists are mentioned
            if self.artist_list_manager:
                poll_text = f"{item.get('theme', '')} {item.get('prompt', '')} {item.get('option_a', '')} {item.get('option_b', '')}".lower()
                if num_options == 4:
                    poll_text += f" {item.get('option_c', '')} {item.get('option_d', '')}".lower()
                uploaded_artists = self.artist_list_manager.get_artist_names_for_prompt()
                if uploaded_artists:
                    artist_mentioned = False
                    for artist in uploaded_artists:
                        if artist.lower() in poll_text:
                            artist_mentioned = True
                            break
                    
                    if not artist_mentioned:
                        continue
            
            # Clean and validate each field
            theme = self._clean_text(item.get('theme', '')).strip() or "Music Poll"
            prompt = self._clean_text(item.get('prompt', '')).strip() or "Which do you prefer?"
            option_a = self._clean_text(item.get('option_a', '')).strip() or "Option A"
            option_b = self._clean_text(item.get('option_b', '')).strip() or "Option B"
            
            # Handle additional options for 4-option polls
            if num_options == 4:
                option_c = self._clean_text(item.get('option_c', '')).strip() or "Option C"
                option_d = self._clean_text(item.get('option_d', '')).strip() or "Option D"
            else:
                option_c = ""
                option_d = ""
            
            # Validate and clean predicted split
            split = self._clean_predicted_split(item.get('predicted_split'), num_options)
            
            # Validate balance score
            try:
                # Slightly lower threshold for 4-option polls since perfect balance is harder
                min_balance = 80.0 if num_options == 4 else 85.0
                balance = max(min_balance, min(98.0, float(item.get('balance_score', 95.0))))
            except (ValueError, TypeError):
                balance = 95.0 if num_options == 2 else 90.0
            
            # Validate predicted score
            try:
                pscore = max(70.0, min(95.0, round(float(item.get('predicted_score', 80.0)), 1)))
            except (ValueError, TypeError):
                pscore = 80.0
            
            best_opt = self._clean_text(item.get('best_balanced_option', '')).strip() or "Option A"
            alternates = item.get('alternates', '')
            if isinstance(alternates, list):
                alternates = '; '.join([self._clean_text(str(x)) for x in alternates])
            else:
                alternates = self._clean_text(str(alternates))
            
            # Get difficulty from item or use default
            item_difficulty = item.get('difficulty', difficulty)
            
            poll_item = {
                'theme': theme,
                'prompt': prompt,
                'option_a': option_a,
                'option_b': option_b,
                'predicted_split': split,
                'balance_score': round(balance, 1),
                'best_balanced_option': best_opt,
                'alternates': alternates,
                'predicted_score': pscore,
                'difficulty': item_difficulty,
                'num_options': num_options  # Store the option count
            }
            
            # Add additional options for 4-option polls
            if num_options == 4:
                poll_item['option_c'] = option_c
                poll_item['option_d'] = option_d
            
            normalized.append(poll_item)
        
        return normalized

    def _clean_text(self, text: str) -> str:
        """Clean text by removing gibberish and random characters"""
        if not text:
            return ""
        
        # Remove common gibberish patterns
        text = re.sub(r'[�◆●✔✅👆🎵🗳️]', '', str(text))  # Remove emojis and special chars
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s?.!,-]', '', text)  # Remove unusual characters but keep basic punctuation
        
        # Fix common issues
        text = re.sub(r'\b(Quantum|Quatum|Quantam)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Optima|Navigation|Practices|Engine)\b', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _clean_predicted_split(self, split_text: Any, num_options: int) -> str:
        """Clean and validate predicted split format for 2 or 4 options"""
        if not split_text:
            if num_options == 2:
                return "50% A / 50% B"
            else:
                return "25% A / 25% B / 25% C / 25% D"
        
        split_str = str(split_text).strip()
        
        # Check if it already has a valid format
        if num_options == 2:
            if re.match(r'^\d+% A / \d+% B$', split_str):
                return split_str
        else:
            if re.match(r'^\d+% A / \d+% B / \d+% C / \d+% D$', split_str):
                return split_str
        
        # Try to extract percentages
        percentages = re.findall(r'\d+', split_str)
        
        if num_options == 2:
            if len(percentages) >= 2:
                a = min(100, max(40, int(percentages[0])))  # Ensure between 40-60
                b = 100 - a
                return f"{a}% A / {b}% B"
            else:
                return "50% A / 50% B"
        else:  # 4 options
            if len(percentages) >= 4:
                # Ensure they sum to 100 and are reasonable
                total = sum(int(p) for p in percentages[:4])
                if total > 0:
                    normalized = [int((int(p) * 100) / total) for p in percentages[:4]]
                    # Adjust to ensure they sum to 100
                    diff = 100 - sum(normalized)
                    if diff > 0:
                        normalized[0] += diff
                    return f"{normalized[0]}% A / {normalized[1]}% B / {normalized[2]}% C / {normalized[3]}% D"
            # Fallback to balanced split
            return "25% A / 25% B / 25% C / 25% D"

    def _extract_clean_example_polls_from_query(self, user_query: str) -> List[Dict[str, str]]:
        """
        Extract and clean example polls from user query with improved natural language processing
        Supports conversational descriptions of poll examples
        """
        if not user_query:
            return []
        
        examples = []
        
        # Pattern 1: Numbered list with "vs" or "or" comparisons
        # Example: "1. Drake vs The Weeknd" or "1. Which fits your mood? Drake or The Weeknd"
        numbered_vs_pattern = r'(\d+)\.\s*(?:([^?]+?)\?)?\s*(?:([^:?]+?)\s*(?:vs\.?|versus|or)\s*([^.\n]+))'
        matches = re.findall(numbered_vs_pattern, user_query, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            question = match[1].strip() if match[1] else "Which do you prefer?"
            option_a = self._clean_text(match[2]).strip()
            option_b = self._clean_text(match[3]).strip()
            
            if option_a and option_b and len(option_a) > 1 and len(option_b) > 1:
                examples.append({
                    'theme': self._infer_theme(option_a, option_b),
                    'prompt': question if question else f"Which do you prefer?",
                    'option_a': option_a.title() if len(option_a.split()) <= 3 else option_a,
                    'option_b': option_b.title() if len(option_b.split()) <= 3 else option_b,
                    'predicted_split': '50% A / 50% B',
                    'balance_score': 95.0
                })
        
        # Pattern 2: Question format with options
        # Example: "Which artist fits your mood better? Option A: Drake, Option B: The Weeknd"
        question_with_options = r'([^?]+\?)\s*(?:option\s*a|a)[:\s]*([^,.\n]+?)[\s,]+(?:option\s*b|b)[:\s]*([^,.\n]+)'
        matches = re.findall(question_with_options, user_query, re.IGNORECASE)
        
        for match in matches:
            question = self._clean_text(match[0].strip() + "?")
            option_a = self._clean_text(match[1]).strip()
            option_b = self._clean_text(match[2]).strip()
            
            if question and option_a and option_b:
                examples.append({
                    'theme': self._infer_theme(option_a, option_b),
                    'prompt': question,
                    'option_a': option_a.title() if len(option_a.split()) <= 3 else option_a,
                    'option_b': option_b.title() if len(option_b.split()) <= 3 else option_b,
                    'predicted_split': '50% A / 50% B',
                    'balance_score': 95.0
                })
        
        # Pattern 3: Simple "A vs B" or "A or B" anywhere in text
        simple_vs_pattern = r'(?:^|\n|\.)\s*([^.\n]+?)\s+(?:vs\.?|versus|or)\s+([^.\n]+?)(?:\.|$|\n)'
        matches = re.findall(simple_vs_pattern, user_query, re.IGNORECASE)
        
        for match in matches:
            option_a = self._clean_text(match[0]).strip()
            option_b = self._clean_text(match[1]).strip()
            
            # Skip if it's part of a larger sentence (too many words)
            if (option_a and option_b and 
                len(option_a.split()) <= 5 and len(option_b.split()) <= 5 and
                len(option_a) > 2 and len(option_b) > 2):
                examples.append({
                    'theme': self._infer_theme(option_a, option_b),
                    'prompt': 'Which do you prefer?',
                    'option_a': option_a.title() if len(option_a.split()) <= 3 else option_a,
                    'option_b': option_b.title() if len(option_b.split()) <= 3 else option_b,
                    'predicted_split': '50% A / 50% B',
                    'balance_score': 95.0
                })
        
        # Pattern 4: Conversational style "like: X or Y"
        like_pattern = r'(?:like|such as|example)[:\s]+([^,.\n]+?)\s+(?:or|vs)\s+([^,.\n]+)'
        matches = re.findall(like_pattern, user_query, re.IGNORECASE)
        
        for match in matches:
            option_a = self._clean_text(match[0]).strip()
            option_b = self._clean_text(match[1]).strip()
            
            if option_a and option_b and len(option_a) > 2 and len(option_b) > 2:
                examples.append({
                    'theme': self._infer_theme(option_a, option_b),
                    'prompt': 'Which resonates more with you?',
                    'option_a': option_a.title() if len(option_a.split()) <= 3 else option_a,
                    'option_b': option_b.title() if len(option_b.split()) <= 3 else option_b,
                    'predicted_split': '50% A / 50% B',
                    'balance_score': 95.0
                })
        
        # Remove duplicates and limit to 5 examples
        seen = set()
        unique_examples = []
        for ex in examples:
            key = (ex['option_a'].lower(), ex['option_b'].lower())
            if key not in seen and len(unique_examples) < 5:
                seen.add(key)
                unique_examples.append(ex)
        
        return unique_examples

    def _infer_theme(self, option_a: str, option_b: str) -> str:
        """Infer poll theme from the options"""
        combined = (option_a + " " + option_b).lower()
        
        if any(word in combined for word in ['artist', 'drake', 'kendrick', 'beyonce', 'taylor', 'singer', 'rapper']):
            return "Artist Comparison"
        elif any(word in combined for word in ['song', 'track', 'hit', 'album', 'music']):
            return "Song Battle"
        elif any(word in combined for word in ['90s', '80s', '2000s', '2010s', 'era', 'decade']):
            return "Era Preference"
        elif any(word in combined for word in ['hip-hop', 'rap', 'pop', 'r&b', 'edm', 'rock', 'genre']):
            return "Genre Debate"
        elif any(word in combined for word in ['vibe', 'mood', 'energy', 'chill', 'hype']):
            return "Vibe Match"
        elif any(word in combined for word in ['workout', 'study', 'morning', 'night', 'party']):
            return "Mood Matching"
        else:
            return "Music Preference"

    def _build_context_from_postgres(self) -> str:
        """Build context string from PostgreSQL data"""
        if not self.db_engine:
            return "No analytics data available"
        
        try:
            context_parts = []
            
            # Get poll performance data from PostgreSQL
            sql_query = """
            SELECT 
                AVG(content_score) as avg_score,
                AVG(engagement_rate_impr) as avg_engagement,
                COUNT(*) as total_posts
            FROM instagram_data 
            WHERE content_score > 0
            """
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
            
            if not df.empty:
                avg_score = df.iloc[0]['avg_score'] or 0
                avg_engagement = df.iloc[0]['avg_engagement'] or 0
                
                context_parts.append(f"Average content performance: {avg_score:.1f}")
                context_parts.append(f"Average engagement rate: {avg_engagement:.2f}%")
            
            # Get successful poll themes from content with high engagement
            themes_query = """
            SELECT hook_type, content_type_tags, COUNT(*) as count
            FROM instagram_data 
            WHERE engagement_rate_impr > 5.0
            AND (hook_type LIKE '%Question%' OR content_type_tags LIKE '%debate%' OR content_type_tags LIKE '%poll%')
            GROUP BY hook_type, content_type_tags
            ORDER BY count DESC
            LIMIT 5
            """
            
            with self.db_engine.connect() as conn:
                themes_df = pd.read_sql(text(themes_query), conn)
            
            if not themes_df.empty:
                top_themes = []
                for _, row in themes_df.iterrows():
                    if row['hook_type'] and row['hook_type'] != 'Unlabeled':
                        top_themes.append(row['hook_type'])
                    elif row['content_type_tags']:
                        # Extract first tag as theme
                        first_tag = row['content_type_tags'].split(';')[0].strip()
                        if first_tag:
                            top_themes.append(first_tag)
                
                if top_themes:
                    context_parts.append(f"Top performing themes: {', '.join(top_themes[:3])}")
            
            balance_query = """
            SELECT COUNT(*) as balanced_count
            FROM instagram_data 
            WHERE engagement_rate_impr > 7.0
            AND (description LIKE '%vs%' OR description LIKE '%or%' OR description LIKE '%debate%')
            """
            
            with self.db_engine.connect() as conn:
                balance_df = pd.read_sql(text(balance_query), conn)
            
            if not balance_df.empty:
                balanced_count = balance_df.iloc[0]['balanced_count'] or 0
                context_parts.append(f"Balanced debate content: {balanced_count} examples")
            
            return " | ".join(context_parts) if context_parts else "Limited poll performance data available"
            
        except Exception as e:
            print(f"Error building context from PostgreSQL: {e}")
            return "Limited poll performance data available"

    def _generate_2option_poll(self, theme: str, difficulty: str, artist1: str, artist2: str) -> tuple:
        """Generate 2-option poll content"""
        if difficulty == "easy":
            if theme == "Artist Comparison":
                prompt = "Who's the more popular artist right now?"
                option_a = artist1
                option_b = artist2
            elif theme == "Song Battle":
                prompt = "Which recent hit is catchier?"
                option_a = f"{artist1} - Popular Song"
                option_b = f"{artist2} - Hit Track"
            else:
                prompt = "Which do you prefer?"
                option_a = artist1
                option_b = artist2
        elif difficulty == "hard":
            if theme == "Artist Comparison":
                prompt = "Whose lyrical craftsmanship is more impressive?"
                option_a = f"{artist1}'s wordplay"
                option_b = f"{artist2}'s storytelling"
            elif theme == "Song Battle":
                prompt = "Which deep cut deserves more recognition?"
                option_a = f"{artist1} - Obscure Track"
                option_b = f"{artist2} - Hidden Gem"
            else:
                prompt = "Which artistic direction is more innovative?"
                option_a = f"{artist1}'s experimental side"
                option_b = f"{artist2}'s creative evolution"
        else:  # medium
            if theme == "Artist Comparison":
                prompt = "Who's the better artist?"
                option_a = artist1
                option_b = artist2
            elif theme == "Song Battle":
                prompt = "Which track hits harder?"
                option_a = f"{artist1} - Hit Song"
                option_b = f"{artist2} - Popular Track"
            elif theme == "Era Preference":
                prompt = "Which era was more influential?"
                option_a = f"{artist1}'s Early Work"
                option_b = f"{artist2}'s Recent Albums"
            elif theme == "Album Showdown":
                prompt = "Which album is a classic?"
                option_a = f"{artist1} - Album One"
                option_b = f"{artist2} - Album Two"
            else:  # Producer Faceoff
                prompt = "Who's the better producer?"
                option_a = f"{artist1}'s Production"
                option_b = f"{artist2}'s Beats"
        
        return prompt, option_a, option_b

    def _generate_4option_poll(self, theme: str, difficulty: str, artist1: str, artist2: str, artist3: str, artist4: str) -> tuple:
        """Generate 4-option poll content"""
        if difficulty == "easy":
            if theme == "Artist Comparison":
                prompt = "Which artist's vibe matches your current mood?"
                option_a = f"{artist1} - Chill vibes"
                option_b = f"{artist2} - Energetic"
                option_c = f"{artist3} - Romantic"
                option_d = f"{artist4} - Party mode"
            elif theme == "Song Battle":
                prompt = "Which genre are you feeling today?"
                option_a = f"{artist1} - Hip Hop"
                option_b = f"{artist2} - R&B"
                option_c = f"{artist3} - Pop"
                option_d = f"{artist4} - Electronic"
            else:
                prompt = "Which music era do you prefer?"
                option_a = f"{artist1} - 90s"
                option_b = f"{artist2} - 2000s"
                option_c = f"{artist3} - 2010s"
                option_d = f"{artist4} - Current"
        elif difficulty == "hard":
            if theme == "Artist Comparison":
                prompt = "Which artist's technical skill impresses you most?"
                option_a = f"{artist1} - Lyrical depth"
                option_b = f"{artist2} - Vocal range"
                option_c = f"{artist3} - Production skills"
                option_d = f"{artist4} - Stage presence"
            elif theme == "Song Battle":
                prompt = "Which production style is most innovative?"
                option_a = f"{artist1} - Experimental"
                option_b = f"{artist2} - Minimalist"
                option_c = f"{artist3} - Orchestral"
                option_d = f"{artist4} - Electronic fusion"
            else:
                prompt = "Which artistic evolution was most impressive?"
                option_a = f"{artist1}'s early to late career"
                option_b = f"{artist2}'s genre shifts"
                option_c = f"{artist3}'s vocal development"
                option_d = f"{artist4}'s production evolution"
        else:  # medium
            if theme == "Artist Comparison":
                prompt = "Which artist would you see live?"
                option_a = artist1
                option_b = artist2
                option_c = artist3
                option_d = artist4
            elif theme == "Song Battle":
                prompt = "Which track would you add to your playlist?"
                option_a = f"{artist1} - Hit Song"
                option_b = f"{artist2} - Popular Track"
                option_c = f"{artist3} - Fan Favorite"
                option_d = f"{artist4} - Deep Cut"
            elif theme == "Era Preference":
                prompt = "Which music decade was the best?"
                option_a = f"{artist1} - 80s Classics"
                option_b = f"{artist2} - 90s Golden Era"
                option_c = f"{artist3} - 2000s Bangers"
                option_d = f"{artist4} - 2010s Hits"
            elif theme == "Album Showdown":
                prompt = "Which album deserves classic status?"
                option_a = f"{artist1} - Debut Album"
                option_b = f"{artist2} - Breakthrough"
                option_c = f"{artist3} - Critical Hit"
                option_d = f"{artist4} - Fan Favorite"
            else:  # Genre Preference
                prompt = "Which genre speaks to you most?"
                option_a = f"{artist1} - Hip Hop/Rap"
                option_b = f"{artist2} - R&B/Soul"
                option_c = f"{artist3} - Pop"
                option_d = f"{artist4} - Alternative"
        
        return prompt, option_a, option_b, option_c, option_d


    def _create_clean_fallback_polls(self, batch_size: int, difficulty: str = "medium", num_options: int = 2) -> pd.DataFrame:
        """Create clean, professional fallback poll data with difficulty levels and option count"""
        fallback_data = []
        themes = ["Artist Comparison", "Song Battle", "Era Preference", "Album Showdown", "Producer Faceoff"]
        
        # Get artists from uploaded list if available
        artists = []
        if self.artist_list_manager:
            artists = self.artist_list_manager.get_artist_names_for_prompt()
        
        # Adjust artist selection based on difficulty
        if artists:
            if difficulty == "easy":
                # Use only the most popular/mainstream artists
                selected_artists = artists[:min(10, len(artists))]
            elif difficulty == "hard":
                # Use more obscure/niche artists
                selected_artists = artists[min(5, len(artists)-10):] if len(artists) > 15 else artists
            else:
                # Medium difficulty - use mixed artists
                selected_artists = artists
        
        for i in range(batch_size):
            theme = themes[i % len(themes)]
            
            if artists and len(selected_artists) >= (4 if num_options == 4 else 2):
                if num_options == 2:
                    artist1 = selected_artists[i % len(selected_artists)]
                    artist2 = selected_artists[(i + 1) % len(selected_artists)]
                else:
                    artist1 = selected_artists[i % len(selected_artists)]
                    artist2 = selected_artists[(i + 1) % len(selected_artists)]
                    artist3 = selected_artists[(i + 2) % len(selected_artists)]
                    artist4 = selected_artists[(i + 3) % len(selected_artists)]
            else:
                # Use generic artists
                if num_options == 2:
                    artist1, artist2 = "Artist A", "Artist B"
                else:
                    artist1, artist2, artist3, artist4 = "Artist A", "Artist B", "Artist C", "Artist D"
            
            # Generate poll content based on difficulty and option count
            if num_options == 2:
                prompt, option_a, option_b = self._generate_2option_poll(theme, difficulty, artist1, artist2)
            else:
                prompt, option_a, option_b, option_c, option_d = self._generate_4option_poll(theme, difficulty, artist1, artist2, artist3, artist4)
            
            # Generate appropriate splits
            if num_options == 2:
                split_a = random.randint(45, 55)
                split_b = 100 - split_a
                predicted_split = f"{split_a}% A / {split_b}% B"
            else:
                splits = [random.randint(15, 35) for _ in range(4)]
                total = sum(splits)
                # Normalize to 100%
                splits = [int((s * 100) / total) for s in splits]
                # Adjust for rounding
                diff = 100 - sum(splits)
                if diff > 0:
                    splits[0] += diff
                predicted_split = f"{splits[0]}% A / {splits[1]}% B / {splits[2]}% C / {splits[3]}% D"
            
            # Adjust scores based on difficulty and option count
            if difficulty == "easy":
                base_score = round(random.uniform(78, 92), 1)
                balance_score = round(random.uniform(90, 97), 1)
            elif difficulty == "hard":
                base_score = round(random.uniform(70, 85), 1)
                balance_score = round(random.uniform(85, 95), 1)  # Slightly lower for hard
            else:
                base_score = round(random.uniform(75, 90), 1)
                balance_score = round(random.uniform(88, 96), 1)
            
            # Slightly adjust balance for 4-option polls (harder to balance perfectly)
            if num_options == 4:
                balance_score = max(85, balance_score - random.uniform(2, 5))
            
            poll_data = {
                "theme": theme,
                "prompt": prompt,
                "option_a": option_a,
                "option_b": option_b,
                "predicted_split": predicted_split,
                "balance_score": round(balance_score, 1),
                "predicted_score": base_score,
                "difficulty": difficulty,
                "num_options": num_options
            }
            
            if num_options == 4:
                poll_data["option_c"] = option_c
                poll_data["option_d"] = option_d
            
            fallback_data.append(poll_data)
        
        return pd.DataFrame(fallback_data)
