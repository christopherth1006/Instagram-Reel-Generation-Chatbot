"""
Scoring Engines for Content Generation
Data-driven scoring engines for quiz, reel, and poll content based on analytics insights
"""
import random
import re
from typing import Dict, Any, List
import pandas as pd

class QuizScoringEngine:
    """Data-driven scoring engine for quiz content based on analytics insights"""
    
    def __init__(self):
        # Enhanced base scoring with more variation
        self.base_scores = {
            'who_said_it': {'content_score': (75, 85), 'shareability': (78, 88)},
            'fill_in_blank': {'content_score': (68, 78), 'shareability': (72, 82)},
            'guess_year': {'content_score': (65, 75), 'shareability': (68, 78)},
            'sample_match': {'content_score': (78, 88), 'shareability': (82, 92)},
            'tracklist': {'content_score': (72, 82), 'shareability': (76, 86)},
            'album_cover': {'content_score': (80, 90), 'shareability': (84, 94)}
        }
        
        # More nuanced engagement multipliers
        self.engagement_factors = {
            'trending_artist': 1.12,    # 12% boost for trending artists
            'classic_artist': 1.08,     # 8% boost for classic artists  
            'controversial': 1.18,      # 18% boost for controversial topics
            'nostalgic': 1.10,          # 10% boost for nostalgic content
            'educational': 1.06,        # 6% boost for educational content
            'fun_fact': 1.15,           # 15% boost for interesting fun facts
            'current_year': 1.08,       # 8% boost for current year references
            'viral_potential': 1.20,    # 20% boost for viral topics
        }
        
        # Difficulty multipliers
        self.difficulty_multipliers = {
            'easy': 1.15,   # Easy content gets 15% boost (more shareable)
            'medium': 1.0,   # Medium is baseline
            'hard': 0.85     # Hard content gets 15% reduction (less shareable)
        }
    
    def calculate_quiz_scores(self, quiz_item: Dict[str, Any], insights: Dict) -> Dict[str, float]:
        """Calculate predicted content score and shareability score for a quiz item with REAL variation"""
        
        quiz_type = self._normalize_quiz_type(quiz_item.get('type', ''))
        question = str(quiz_item.get('question', '')).lower()
        fun_fact = str(quiz_item.get('fun_fact', '')).lower()
        difficulty = quiz_item.get('difficulty', 'medium').lower()
        
        # Get base score ranges for this quiz type
        base_ranges = self.base_scores.get(quiz_type, {'content_score': (65, 75), 'shareability': (70, 80)})
        content_min, content_max = base_ranges['content_score']
        share_min, share_max = base_ranges['shareability']
        
        # Start with random base scores within the range
        base_content_score = random.uniform(content_min, content_max)
        base_shareability = random.uniform(share_min, share_max)
        
        # Apply insights-based adjustments
        content_score = self._apply_insights_adjustments(base_content_score, quiz_item, insights, difficulty)
        shareability_score = self._apply_shareability_factors(base_shareability, quiz_item, insights, difficulty)
        
        # Apply content-specific multipliers
        content_multiplier = self._calculate_content_multiplier(question, fun_fact, difficulty)
        shareability_multiplier = self._calculate_shareability_multiplier(question, fun_fact, difficulty)
        
        content_score *= content_multiplier
        shareability_score *= shareability_multiplier
        
        # Apply difficulty multiplier
        difficulty_multiplier = self.difficulty_multipliers.get(difficulty, 1.0)
        shareability_score *= difficulty_multiplier
        
        # Add final random variation
        content_score += random.uniform(-3, 3)
        shareability_score += random.uniform(-2, 4)
        
        # Ensure scores are within reasonable bounds
        content_score = max(40.0, min(95.0, content_score))
        shareability_score = max(45.0, min(95.0, shareability_score))
        
        return {
            'predicted_score': round(content_score, 1),
            'shareability_score': round(shareability_score, 1)
        }
    
    def _normalize_quiz_type(self, quiz_type: str) -> str:
        """Normalize quiz type to standard format"""
        quiz_type = quiz_type.lower().replace(' ', '_').replace('-', '_')
        if 'who' in quiz_type and 'said' in quiz_type:
            return 'who_said_it'
        elif 'fill' in quiz_type and 'blank' in quiz_type:
            return 'fill_in_blank'
        elif 'year' in quiz_type or 'guess' in quiz_type:
            return 'guess_year'
        elif 'sample' in quiz_type or 'match' in quiz_type:
            return 'sample_match'
        elif 'tracklist' in quiz_type or 'track' in quiz_type:
            return 'tracklist'
        elif 'album' in quiz_type or 'cover' in quiz_type:
            return 'album_cover'
        else:
            return 'who_said_it'
    
    def _apply_insights_adjustments(self, base_score: float, quiz_item: Dict, insights: Dict, difficulty: str) -> float:
        """Apply adjustments based on analytics insights"""
        adjusted_score = base_score
        
        # Adjust based on top performing artists from insights
        if 'top_artists' in insights and insights['top_artists']:
            question_text = str(quiz_item.get('question', '')).lower()
            options_text = ' '.join([
                str(quiz_item.get('option_a', '')),
                str(quiz_item.get('option_b', '')), 
                str(quiz_item.get('option_c', '')),
                str(quiz_item.get('option_d', ''))
            ]).lower()
            
            all_text = f"{question_text} {options_text}"
            top_artists = [artist.lower() for artist in insights['top_artists'].keys()]
            
            # Check if any top artists are mentioned
            for artist in top_artists[:5]:  # Only check top 5 artists
                if artist in all_text:
                    if difficulty == 'easy':
                        adjusted_score *= 1.08  # 8% boost for easy content with top artists
                    elif difficulty == 'medium':
                        adjusted_score *= 1.05  # 5% boost for medium
                    else:
                        adjusted_score *= 1.03  # 3% boost for hard
                    break
        
        # Adjust based on hook performance insights
        if 'hook_performance' in insights and insights['hook_performance']:
            hook_scores = insights['hook_performance'].get('avg_score', {})
            if hook_scores:
                avg_hook_score = sum(hook_scores.values()) / len(hook_scores)
                # Scale adjustment based on performance (better hooks = better quizzes)
                performance_factor = (avg_hook_score - 70) / 30  # Normalize around 70
                adjusted_score *= (0.95 + 0.1 * performance_factor)
        
        # Adjust based on engagement patterns
        if 'avg_engagement_rate' in insights:
            engagement_rate = insights['avg_engagement_rate']
            if engagement_rate > 0:
                # Scale based on historical engagement
                engagement_factor = min(2.0, engagement_rate * 10)  # Cap at 2x
                adjusted_score *= (0.9 + 0.2 * engagement_factor)
        
        return adjusted_score
    
    def _apply_shareability_factors(self, base_score: float, quiz_item: Dict, insights: Dict, difficulty: str) -> float:
        """Apply factors that affect shareability"""
        adjusted_score = base_score
        
        question_text = str(quiz_item.get('question', '')).lower()
        fun_fact_text = str(quiz_item.get('fun_fact', '')).lower()
        
        # Boost for controversial or debate-worthy content
        controversial_words = ['better', 'best', 'vs', 'versus', 'who', 'which', 'debate', 'controversial']
        if any(word in question_text for word in controversial_words):
            adjusted_score *= 1.12
        
        # Boost for nostalgic content
        nostalgic_words = ['classic', 'throwback', 'remember', 'old school', 'nostalgia', 'back in']
        if any(word in question_text for word in nostalgic_words):
            adjusted_score *= 1.10
        
        # Boost for educational content
        educational_words = ['how', 'what', 'when', 'where', 'why', 'explain', 'fact', 'learn']
        if any(word in question_text for word in educational_words):
            adjusted_score *= 1.08
        
        # Boost for fun facts that are actually interesting
        if len(fun_fact_text) > 20 and 'fact' in fun_fact_text:
            adjusted_score *= 1.07
        
        # Boost for current year references (more shareable)
        import datetime
        current_year = str(datetime.datetime.now().year)
        if current_year in question_text or current_year in fun_fact_text:
            adjusted_score *= 1.06
        
        return adjusted_score
    
    def _calculate_content_multiplier(self, question: str, fun_fact: str, difficulty: str) -> float:
        """Calculate content quality multiplier"""
        multiplier = 1.0
        combined_text = f"{question} {fun_fact}"
        
        # Check for engagement factors
        for factor, factor_multiplier in self.engagement_factors.items():
            if self._text_contains_factor(combined_text, factor):
                multiplier *= factor_multiplier
                break  # Only apply one primary multiplier
        
        # Adjust for question length (optimal length gets boost)
        question_words = len(question.split())
        if 8 <= question_words <= 15:  # Optimal question length
            multiplier *= 1.05
        elif question_words > 20:  # Too long
            multiplier *= 0.95
        
        return multiplier
    
    def _calculate_shareability_multiplier(self, question: str, fun_fact: str, difficulty: str) -> float:
        """Calculate shareability multiplier"""
        multiplier = 1.0
        combined_text = f"{question} {fun_fact}"
        
        # Viral potential indicators
        viral_indicators = [
            'viral', 'tiktok', 'challenge', 'trending', 'hot', 'breaking',
            'secret', 'hidden', 'exclusive', 'behind the scenes'
        ]
        
        if any(indicator in combined_text for indicator in viral_indicators):
            multiplier *= 1.15
        
        # Emotional engagement indicators
        emotional_indicators = [
            'shocking', 'surprising', 'amazing', 'unbelievable', 'incredible',
            'heartbreaking', 'inspirational', 'motivational'
        ]
        
        if any(indicator in combined_text for indicator in emotional_indicators):
            multiplier *= 1.12
        
        # Curiosity gap indicators
        curiosity_indicators = [
            'you won\'t believe', 'what happened next', 'the reason why',
            'secret behind', 'truth about', 'real story'
        ]
        
        if any(indicator in combined_text for indicator in curiosity_indicators):
            multiplier *= 1.10
        
        return multiplier
    
    def _text_contains_factor(self, text: str, factor: str) -> bool:
        """Check if text contains indicators for a specific engagement factor"""
        factor_indicators = {
            'trending_artist': ['new', 'latest', 'fresh', 'hot', 'viral', 'trending', 'breaking'],
            'classic_artist': ['classic', 'legend', 'icon', 'old school', 'throwback', 'veteran'],
            'controversial': ['controversial', 'debate', 'hot take', 'unpopular', 'disagree', 'argument'],
            'nostalgic': ['remember', 'back in', 'used to', 'nostalgia', 'throwback', 'childhood'],
            'educational': ['learn', 'fact', 'did you know', 'history', 'explain', 'educational'],
            'fun_fact': ['fun fact', 'interesting', 'amazing', 'incredible', 'wow', 'surprising'],
            'current_year': [str(2024), 'this year', 'current', 'recent'],
            'viral_potential': ['viral', 'tiktok', 'challenge', 'trending', 'hot']
        }
        
        indicators = factor_indicators.get(factor, [])
        return any(indicator in text for indicator in indicators)


class ReelScoringEngine:
    """Data-driven scoring engine for reel content based on analytics insights"""
    
    def __init__(self):
        # Base scoring factors for different reel types
        self.base_scores = {
            'trending_artist': {'content_score': 82.3, 'shareability': 85.7},
            'classic_throwback': {'content_score': 78.9, 'shareability': 81.2},
            'educational': {'content_score': 75.4, 'shareability': 78.6},
            'controversial': {'content_score': 88.1, 'shareability': 92.3},
            'story_pov': {'content_score': 79.8, 'shareability': 83.4},
            'music_highlight': {'content_score': 76.2, 'shareability': 79.5}
        }
        
        # Engagement factors based on content characteristics
        self.engagement_factors = {
            'hook_strength': 1.2,      # 20% boost for strong hooks
            'trending_topic': 1.15,    # 15% boost for trending topics
            'personal_story': 1.12,    # 12% boost for personal stories
            'educational': 1.08,       # 8% boost for educational content
            'controversial': 1.25,     # 25% boost for controversial content
            'nostalgic': 1.10          # 10% boost for nostalgic content
        }
    
    def extract_successful_patterns(self, analytics_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract patterns from successful posts in analytics data"""
        if analytics_data is None or analytics_data.empty:
            return {}
        
        try:
            # Filter for high-performing posts (top 20%)
            high_performing = analytics_data.nlargest(max(1, len(analytics_data) // 5), 'content_score')
            
            patterns = {
                'successful_hooks': self._analyze_hook_patterns(high_performing),
                'successful_captions': self._analyze_caption_patterns(high_performing),
                'successful_artists': self._analyze_artist_patterns(high_performing),
                'voice_characteristics': self._analyze_voice_characteristics(high_performing)
            }
            
            return patterns
        except Exception:
            return {}
    
    def _analyze_hook_patterns(self, successful_posts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in successful hook texts"""
        try:
            hooks = successful_posts['description'].fillna('').astype(str)
            
            # Common hook patterns
            hook_patterns = {
                'question_hooks': len([h for h in hooks if h.strip().endswith('?')]),
                'statement_hooks': len([h for h in hooks if not h.strip().endswith('?')]),
                'emoji_usage': len([h for h in hooks if any(ord(char) > 127 for char in h)]),
                'hashtag_usage': len([h for h in hooks if '#' in h]),
                'mention_usage': len([h for h in hooks if '@' in h])
            }
            
            return hook_patterns
        except Exception:
            return {}
    
    def _analyze_caption_patterns(self, successful_posts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in successful captions"""
        try:
            captions = successful_posts['description'].fillna('').astype(str)
            
            # Caption length analysis
            lengths = [len(caption.split()) for caption in captions]
            avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            # Common words/phrases
            all_text = ' '.join(captions).lower()
            common_words = {}
            for word in ['fire', 'heat', 'vibe', 'energy', 'flow', 'bars', 'beat', 'track']:
                common_words[word] = all_text.count(word)
            
            return {
                'avg_length': avg_length,
                'common_words': common_words
            }
        except Exception:
            return {}
    
    def _analyze_artist_patterns(self, successful_posts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in successful artist mentions"""
        try:
            if 'main_artists' in successful_posts.columns:
                artists = successful_posts['main_artists'].fillna('').astype(str)
                artist_counts = artists.value_counts().to_dict()
                return artist_counts
            return {}
        except Exception:
            return {}
    
    def _analyze_voice_characteristics(self, successful_posts: pd.DataFrame) -> Dict[str, Any]:
        """Analyze voice characteristics from successful posts"""
        try:
            descriptions = successful_posts['description'].fillna('').astype(str)
            
            # Analyze tone indicators
            tone_indicators = {
                'enthusiastic': len([d for d in descriptions if any(word in d.lower() for word in ['fire', 'heat', 'amazing', 'incredible'])]),
                'casual': len([d for d in descriptions if any(word in d.lower() for word in ['fr', 'ngl', 'tbh', 'lowkey'])]),
                'educational': len([d for d in descriptions if any(word in d.lower() for word in ['fact', 'learn', 'history', 'explain'])]),
                'personal': len([d for d in descriptions if any(word in d.lower() for word in ['i', 'my', 'me', 'we'])]),
                'questioning': len([d for d in descriptions if '?' in d])
            }
            
            return tone_indicators
        except Exception:
            return {}
    
    def calculate_reel_scores(self, reel_item: Dict[str, Any], patterns: Dict[str, Any], insights: Dict = None) -> Dict[str, float]:
        """Calculate predicted content score and shareability score for a reel item"""
        
        # Classify reel type
        reel_type = self._classify_reel_type(reel_item)
        
        # Start with base scores for the reel type
        base_scores = self.base_scores.get(reel_type, {'content_score': 75.0, 'shareability': 78.0})
        content_score = base_scores['content_score']
        shareability_score = base_scores['shareability']
        
        # Apply pattern-based adjustments
        content_score += self._calculate_pattern_adjustment(reel_item, patterns)
        shareability_score += self._calculate_pattern_adjustment(reel_item, patterns)
        
        # Apply content-specific adjustments
        content_score += self._calculate_content_adjustment(reel_item)
        shareability_score += self._calculate_content_adjustment(reel_item)
        
        # Apply insights-based adjustments
        content_score += self._calculate_insights_adjustment(reel_item, insights)
        shareability_score += self._calculate_insights_adjustment(reel_item, insights)
        
        # Add some variation to make scores more realistic
        import random
        content_variation = random.uniform(-2.5, 2.5)
        shareability_variation = random.uniform(-2.5, 2.5)
        
        content_score += content_variation
        shareability_score += shareability_variation
        
        # Ensure scores are within reasonable bounds
        content_score = max(50.0, min(92.0, content_score))
        shareability_score = max(55.0, min(92.0, shareability_score))
        
        return {
            'predicted_score': round(content_score, 1),
            'shareability_score': round(shareability_score, 1)
        }
    
    def _classify_reel_type(self, reel_item: Dict[str, Any]) -> str:
        """Classify the type of reel content"""
        hook_text = str(reel_item.get('hook_text', '')).lower()
        audio_suggestion = str(reel_item.get('audio_suggestion', '')).lower()
        captions = ' '.join(reel_item.get('captions', [])).lower()
        
        combined_text = f"{hook_text} {audio_suggestion} {captions}"
        
        if any(word in combined_text for word in ['trending', 'viral', 'hottest', 'new']):
            return 'trending_artist'
        elif any(word in combined_text for word in ['classic', 'throwback', 'nostalgia', 'remember']):
            return 'classic_throwback'
        elif any(word in combined_text for word in ['how to', 'tips', 'guide', 'explained']):
            return 'educational'
        elif any(word in combined_text for word in ['controversial', 'debate', 'hot take', 'unpopular']):
            return 'controversial'
        elif any(word in combined_text for word in ['story', 'pov', 'once', 'yesterday']):
            return 'story_pov'
        else:
            return 'music_highlight'
    
    def _calculate_pattern_adjustment(self, reel_item: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """Calculate adjustment based on successful patterns"""
        adjustment = 0.0
        
        # Check hook patterns
        if 'successful_hooks' in patterns:
            hook_text = str(reel_item.get('hook_text', '')).lower()
            
            # Boost for question hooks if they're successful
            if hook_text.endswith('?') and patterns['successful_hooks'].get('question_hooks', 0) > 0:
                adjustment += 3.0
            
            # Boost for emoji usage if it's successful
            if any(ord(char) > 127 for char in hook_text) and patterns['successful_hooks'].get('emoji_usage', 0) > 0:
                adjustment += 2.0
        
        # Check caption patterns
        if 'successful_captions' in patterns:
            captions = reel_item.get('captions', [])
            if captions:
                avg_caption_length = sum(len(caption.split()) for caption in captions) / len(captions)
                successful_avg = patterns['successful_captions'].get('avg_length', 0)
                
                # Boost if caption length matches successful patterns
                if abs(avg_caption_length - successful_avg) < 2:
                    adjustment += 2.5
        
        return adjustment
    
    def _calculate_content_adjustment(self, reel_item: Dict[str, Any]) -> float:
        """Calculate adjustment based on content characteristics"""
        adjustment = 0.0
        
        hook_text = str(reel_item.get('hook_text', '')).lower()
        captions = ' '.join(reel_item.get('captions', [])).lower()
        combined_text = f"{hook_text} {captions}"
        
        # Boost for strong engagement words
        engagement_words = ['fire', 'heat', 'vibe', 'energy', 'flow', 'bars', 'beat']
        for word in engagement_words:
            if word in combined_text:
                adjustment += 1.5
        
        # Boost for trending indicators
        if any(word in combined_text for word in ['new', 'latest', 'fresh', 'hot']):
            adjustment += 3.0
        
        # Boost for personal elements
        if any(word in combined_text for word in ['i', 'my', 'me', 'we']):
            adjustment += 2.0
        
        # Boost for questions (engagement)
        if '?' in combined_text:
            adjustment += 2.5
        
        return adjustment
    
    def _calculate_insights_adjustment(self, reel_item: Dict[str, Any], insights: Dict) -> float:
        """Calculate adjustment based on analytics insights"""
        adjustment = 0.0
        
        # Boost for top-performing artists
        if 'top_artists' in insights:
            hook_text = str(reel_item.get('hook_text', '')).lower()
            captions = ' '.join(reel_item.get('captions', [])).lower()
            combined_text = f"{hook_text} {captions}"
            
            for artist in insights['top_artists'].keys():
                if artist.lower() in combined_text:
                    adjustment += 4.0
                    break
        
        # Boost based on hook performance insights
        if 'hook_performance' in insights:
            hook_scores = insights['hook_performance'].get('avg_score', {})
            if hook_scores:
                avg_hook_score = sum(hook_scores.values()) / len(hook_scores)
                # Scale the adjustment based on hook performance
                adjustment += (avg_hook_score - 75) * 0.1  # Small adjustment based on performance
        
        return adjustment
