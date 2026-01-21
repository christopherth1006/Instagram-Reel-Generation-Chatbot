"""
Reel Generation Module - Updated to use PostgreSQL
Handles generation of Instagram Reel scripts with artist filtering and batch size control
"""
import json
import random
from typing import List, Dict, Any, Optional
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from sqlalchemy import text
from .scoring_engines import ReelScoringEngine
import streamlit as st

class ReelGenerator:
    """Handles reel generation logic using PostgreSQL"""
    
    def __init__(self, llm, brand_voice: Dict[str, Any], artist_list_manager=None, db_engine=None):
        self.llm = llm
        self.brand_voice = brand_voice
        self.artist_list_manager = artist_list_manager
        self.db_engine = db_engine  # Use db_engine to match your existing pattern
        self.reel_scoring_engine = ReelScoringEngine()
    
    def generate_reel_scripts(self, 
                          insights: Dict, 
                          batch_size: int,
                          focus_themes: Optional[List[str]] = None,
                          analytics_data: Optional[pd.DataFrame] = None,
                          chat_history: List = []
                          ) -> pd.DataFrame:
        """Generate Reel scripts using data from PostgreSQL, respecting focus_themes"""

        # Extract successful patterns from PostgreSQL
        patterns = self._extract_patterns_from_postgres()
        print(f"patterns: {patterns}")

        # Get allowed artists from uploaded list
        allowed_artists_exact: List[str] = []
        if self.artist_list_manager:
            allowed_artists_exact = self.artist_list_manager.get_artist_names_for_prompt() or []

        # Get analytics data from PostgreSQL for artist ranking
        analytics_allowed_ranked = self._get_artists_from_postgres(allowed_artists_exact)

        # If no analytics artists match, use the uploaded list directly
        if not analytics_allowed_ranked and allowed_artists_exact:
            analytics_allowed_ranked = allowed_artists_exact

        print(f"allowed aritst: {analytics_allowed_ranked}")

        # Filter artists by focus_themes if provided
        if focus_themes and analytics_data is not None:
            filtered_artists = []
            for artist in analytics_allowed_ranked:
                artist_theme = analytics_data.get("theme", {}).get(artist.lower())
                if artist_theme and artist_theme.lower() in [t.lower() for t in focus_themes]:
                    filtered_artists.append(artist)
            if filtered_artists:
                analytics_allowed_ranked = filtered_artists

        # Build context from PostgreSQL insights and focus themes
        context = self._build_context_from_postgres()

        print(f"context: {context}")
        if focus_themes:
            context += f"\n\nFOCUS THEMES FOR THIS BATCH: {', '.join(focus_themes)}"

        # Add artist filtering to context
        artist_filter = ""
        if self.artist_list_manager:
            artist_filter = self.artist_list_manager.get_artist_filter_prompt()
            if artist_filter:
                context += f"\n\nSTRICT ARTIST FILTERING: {artist_filter}"
                context += "\n\nCRITICAL: You MUST ONLY use artists from the provided list. Do not suggest any artists not in the listed list."

        # Generate reels using LLM with PostgreSQL context
        try:
            items = self._generate_reel_items(context, batch_size, chat_history)
            print("reel items: ", items)
            # Post-process to ensure all artists are from the allowed list
            items_to_process = []
            for item in items:
                suggested_artist = item.get('artist', '').strip()
                if suggested_artist and suggested_artist in analytics_allowed_ranked:
                    items_to_process.append(item)
                elif analytics_allowed_ranked:
                    # Replace with a random artist from the allowed pool
                    item['artist'] = random.choice(analytics_allowed_ranked)
                    items_to_process.append(item)

            print("item to process: ", items_to_process, "\nbatch size: ", batch_size)

            # Pad with additional items if needed
            if len(items_to_process) < batch_size:
                remaining = batch_size - len(items_to_process)
                for i in range(remaining):
                    artist = analytics_allowed_ranked[i % len(analytics_allowed_ranked)]
                    items_to_process.append({
                        "artist": artist,
                        "hook_text": f"Why {artist} stays in rotation",
                        "audio_suggestion": f"Top track - {artist}",
                        "captions": [
                            f"{artist} been on repeat fr 🎧",
                            f"Underrated cuts from {artist} you need",
                            f"Lowkey {artist} in a different bag rn"
                        ],
                        "ctas": ["Tap to stream 🔗", "Add to playlist ➕", "Watch the clip ▶️"]
                    })


            # Normalize and score the items
            normalized = []
            for item in items_to_process:
                scores = self.reel_scoring_engine.calculate_reel_scores(item, patterns, st.session_state.insights)
                print("scores: ", scores)
                normalized.append({
                    "artist": item["artist"],
                    "hook_text": item["hook_text"],
                    "audio_suggestion": item["audio_suggestion"],
                    "captions": item["captions"],
                    "ctas": item["ctas"],
                    "predicted_score": scores['predicted_score']
                })

            print("normalized reels items", normalized)

            return pd.DataFrame(normalized)

        except Exception as e:
            print(f"Error generating reels: {e}")
            # Fallback to artist list-based generation
            return self._create_fallback_reels_from_artist_list(batch_size, patterns, st.session_state.insights)   
        
    def _extract_patterns_from_postgres(self) -> Dict[str, Any]:
        """Extract successful patterns from PostgreSQL database"""
        if not self.db_engine:
            return {}
        
        try:
            # Query for high-performing content patterns
            sql_query = """
            SELECT description, content_score, main_artists, hook_type, engagements
            FROM instagram_data 
            WHERE content_score > 0.7 
            ORDER BY content_score DESC 
            LIMIT 50
            """
            
            with self.db_engine.connect() as conn:  # Directly use db_engine
                df = pd.read_sql(text(sql_query), conn)
            
            if df.empty:
                return {}
            
            patterns = {
                'successful_hooks': self._analyze_hook_patterns(df),
                'successful_captions': self._analyze_caption_patterns(df),
                'successful_artists': self._analyze_artist_patterns(df),
                'voice_characteristics': self._analyze_voice_characteristics(df)
            }
            
            return patterns
        except Exception as e:
            print(f"Error extracting patterns from PostgreSQL: {e}")
            return {}
        
    def _get_artists_from_postgres(self, allowed_artists: List[str]) -> List[str]:
        """Get artists from PostgreSQL that match the allowed list"""
        if not self.db_engine or not allowed_artists:
            return []
        
        try:
            # Create SQL condition for allowed artists
            artist_conditions = " OR ".join([f"LOWER(main_artists) LIKE '%{artist.lower()}%'" for artist in allowed_artists])
            
            sql_query = f"""
            SELECT main_artists, content_score, engagements
            FROM instagram_data 
            WHERE ({artist_conditions}) 
            AND content_score > 65
            ORDER BY content_score DESC, engagements DESC
            LIMIT 100
            """
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
            
            if df.empty:
                return []
            
            # Extract and rank artists
            ranked_artists = []
            for _, row in df.iterrows():
                artist_name = row['main_artists']
                if artist_name and artist_name.strip():
                    score = row.get('content_score', 0)
                    if score > 65:  # Only include decent-performing artists
                        ranked_artists.append((artist_name, score))
            
            # Sort by score and return unique artists
            ranked_artists.sort(key=lambda x: x[1], reverse=True)
            return list(dict.fromkeys([artist for artist, score in ranked_artists]))
            
        except Exception as e:
            print(f"Error getting artists from PostgreSQL: {e}")
            return []
        
    def _build_context_from_postgres(self) -> str:
        """Build context string from PostgreSQL data"""
        if not self.db_engine:
            return "No analytics data available"
        
        try:
            context_parts = []
            
            # Get top performing content for context
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
            
            # Get common successful themes
            themes_query = """
            SELECT hook_type, COUNT(*) as count
            FROM instagram_data 
            WHERE content_score > 0.7
            GROUP BY hook_type
            ORDER BY count DESC
            LIMIT 5
            """
            
            with self.db_engine.connect() as conn:
                themes_df = pd.read_sql(text(themes_query), conn)
            
            if not themes_df.empty:
                top_themes = themes_df['hook_type'].tolist()
                context_parts.append(f"Top performing hook types: {', '.join(top_themes)}")
            
            return " | ".join(context_parts) if context_parts else "Limited performance data available"
            
        except Exception as e:
            print(f"Error building context from PostgreSQL: {e}")
            return "Limited performance data available"
        
    def _analyze_hook_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hook patterns from PostgreSQL results"""
        hook_patterns = {
            'question_hooks': 0,
            'statement_hooks': 0,
            'emoji_usage': 0
        }
        
        for description in df['description'].fillna(''):
            if not isinstance(description, str):
                continue
                
            if description.strip().endswith('?'):
                hook_patterns['question_hooks'] += 1
            else:
                hook_patterns['statement_hooks'] += 1
            
            if any(ord(char) > 127 for char in description):
                hook_patterns['emoji_usage'] += 1
        
        return hook_patterns
    
    def _analyze_caption_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze caption patterns from PostgreSQL results"""
        try:
            captions = df['description'].fillna('').astype(str)
            
            lengths = [len(caption.split()) for caption in captions]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            all_text = ' '.join([caption.lower() for caption in captions])
            common_words = {}
            for word in ['fire', 'heat', 'vibe', 'energy', 'flow', 'bars', 'beat', 'track']:
                common_words[word] = all_text.count(word)
            
            return {
                'avg_length': avg_length,
                'common_words': common_words
            }
        except Exception:
            return {'avg_length': 0, 'common_words': {}}
    
    def _analyze_artist_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze artist patterns from PostgreSQL results"""
        try:
            artist_counts = {}
            for artists in df['main_artists'].fillna(''):
                if artists and isinstance(artists, str):
                    artist_counts[artists] = artist_counts.get(artists, 0) + 1
            
            return artist_counts
        except Exception:
            return {}
    
    def _analyze_voice_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze voice characteristics from PostgreSQL results"""
        tone_indicators = {
            'enthusiastic': 0,
            'casual': 0,
            'educational': 0,
            'personal': 0,
            'questioning': 0
        }
        
        for description in df['description'].fillna(''):
            if not isinstance(description, str):
                continue
                
            content = description.lower()
            if any(word in content for word in ['fire', 'heat', 'amazing', 'incredible']):
                tone_indicators['enthusiastic'] += 1
            if any(word in content for word in ['fr', 'ngl', 'tbh', 'lowkey']):
                tone_indicators['casual'] += 1
            if any(word in content for word in ['fact', 'learn', 'history', 'explain']):
                tone_indicators['educational'] += 1
            if any(word in content for word in ['i', 'my', 'me', 'we']):
                tone_indicators['personal'] += 1
            if '?' in content:
                tone_indicators['questioning'] += 1
        
        return tone_indicators
    
    def _generate_reel_items(self, context: str, batch_size: int, chat_history: List) -> List[Dict[str, Any]]:
        """Generate reel items using LLM"""
        system_prompt = f"""
        You are {self.brand_voice.get('persona_name', 'Cam | VibeTherapy')}, a music curator for rap/R&B.
        
        YOUR AUTHENTIC VOICE & PERSONALITY:
        - Tone: {', '.join(self.brand_voice.get('tone', ['confident', 'playful', 'conversational', 'funny', 'crazy']))}
        - Reading level: {self.brand_voice.get('reading_level', 'Casual, 8th - 12th grade')}
        - Your slang: {', '.join(self.brand_voice.get('slang_bank', ['yo', 'crazy', 'lock in']))}
        - Emojis: {self.brand_voice.get('emojis_frequency', 'light')} use of {', '.join(self.brand_voice.get('emojis_bank', ['🏆', '🔥', '👇', '🫴', '😭', '🤣', '💔', '🥀', '✅']))}
        
        CAPTION WRITING STYLE - BE AUTHENTIC:
        1. Write like YOU personally - use "I", "me", "my opinion"
        2. Include your genuine reactions: "this hits different", "I can't stop playing this", "this is crazy"
        3. Use your actual slang naturally: "yo this is fire", "lock in on this one", "this is crazy good"
        4. Share personal takes: "I think this is underrated", "this reminds me of...", "I've been obsessed with..."
        5. Be conversational like talking to friends: "y'all sleeping on this", "trust me on this one"
        6. Don't sound like a robot or generic influencer - sound like YOU
        7. Use contractions: "I'm", "you're", "they're", "can't", "won't"
        8. Be specific about why you like something: "the flow is insane", "the beat goes crazy", "the lyrics hit different"
        
        IMPORTANT: Use DIFFERENT artists for each reel - don't repeat the same artist multiple times
        If the user wants to update the previous generated reels, Update those reels as per user's request, NOT generating a NEW ONE. And indicate you updated the previous ones. 
        
        Performance Context:
        {context}
        """
        
        human_prompt = f"""
        Generate exactly {batch_size} Instagram Reel ideas.
        
        CRITICAL: Use DIFFERENT artists for each of the {batch_size} reels. Don't repeat the same artist multiple times. Vary the artists to create diverse content.
        
        Each reel should have:
        1. A hook_text (opening line that grabs attention - be direct and punchy)
        2. An audio_suggestion (song/track recommendation)
        3. Three captions (write like YOU personally - use "I", "me", your slang, personal opinions, contractions)
        4. Three CTAs (call-to-action buttons)
        
        CAPTION EXAMPLES OF YOUR STYLE:
        - "I can't stop playing this track 🔥 the flow is insane"
        - "Yo this is crazy underrated, y'all sleeping on this one"
        - "This hits different fr, I've been obsessed with this beat"
        - "Trust me on this one, this song goes crazy"
        - "I think this is the best track they've dropped this year"

        If chat_history contains previous reels and the user intends to update them:
        - Edit the existing reels for clarity, engagement, and correctness.
        - Keep original reels types but improve content.
        
        Return as a single JSON array in a fenced code block like:
        ```json
        [{{
            "artist": "Artist Name",
            "hook_text": "Opening line that hooks viewers",
            "audio_suggestion": "Song recommendation",
            "captions": ["Caption 1", "Caption 2", "Caption 3"],
            "ctas": ["CTA 1", "CTA 2", "CTA 3"]
        }}]
        ```
        """
        
        chat_history_clone =  chat_history.copy()

        chat_history_clone.append(HumanMessage(content=system_prompt))
        chat_history_clone.append(HumanMessage(content=human_prompt))
        print(f"chat history: {chat_history}")
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
        
        return json.loads(content)
    
    def _create_fallback_reels_from_artist_list(self, batch_size: int, patterns: Dict = None, insights: Dict = None) -> pd.DataFrame:
        """Create a minimal set of reels directly from the uploaded artist list when strict filtering removes all items."""
        try:
            artists = self.artist_list_manager.get_artist_names_for_prompt() if self.artist_list_manager else []
            if not artists:
                return pd.DataFrame(columns=["artist","hook_text","audio_suggestion","captions","ctas","predicted_score"]) 
            
            import random
            random.shuffle(artists)
            rows: List[Dict[str, Any]] = []
            for idx in range(batch_size):
                artist = artists[idx % len(artists)]  # Cycle through all artists
                reel_item = {
                    "artist": artist,
                    "hook_text": f"Why {artist} stays in rotation",
                    "audio_suggestion": f"Top track - {artist}",
                    "captions": [
                        f"I can't stop playing {artist} fr 🔥 the flow is insane",
                        f"Yo this is crazy underrated, y'all sleeping on {artist}",
                        f"This hits different, I've been obsessed with {artist}'s sound"
                    ],
                    "ctas": [
                        "Tap to stream 🔗",
                        "Add to playlist ➕",
                        "Watch the clip ▶️"
                    ]
                }
                # Score
                if patterns is not None and insights is not None:
                    scores = self.reel_scoring_engine.calculate_reel_scores(reel_item, patterns, insights)
                    pscore = scores['predicted_score']
                else:
                    pscore = round(random.uniform(68, 88), 1)
                rows.append({
                    "artist": reel_item["artist"],
                    "hook_text": reel_item["hook_text"],
                    "audio_suggestion": reel_item["audio_suggestion"],
                    "captions": reel_item["captions"],
                    "ctas": reel_item["ctas"],
                    "predicted_score": pscore
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error in fallback reel generation: {e}")
            return pd.DataFrame(columns=["artist","hook_text","audio_suggestion","captions","ctas","predicted_score"])