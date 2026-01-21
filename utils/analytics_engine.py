"""
Enhanced Analytics Engine for Instagram Content Analysis
Builds on the existing MVP transformer with improved ContentScore calculation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
from config import CONTENT_SCORE_WEIGHTS, MIN_IMPRESSIONS_THRESHOLD, SMOOTHING_FACTOR

class AnalyticsEngine:
    def __init__(self):
        self.content_score_weights = CONTENT_SCORE_WEIGHTS
        self.min_impressions = MIN_IMPRESSIONS_THRESHOLD
        self.smoothing_factor = SMOOTHING_FACTOR
        
        # Enhanced pattern matching from existing code
        self.patterns = {
            'question': re.compile(r"\?\s*$|^\s*(how|what|why|who|when|where|is|are|can|should|do|does|did)\b", re.I),
            'educational': re.compile(r"\b(how to|tutorial|tips|guide|secrets|mistakes|thread|breakdown|explained)\b", re.I),
            'trend': re.compile(r"\b(trend|challenge|viral|capcut|template)\b", re.I),
            'meme': re.compile(r"\b(meme|lol|funny|joke|humor|lmao)\b", re.I),
            'story': re.compile(r"\b(story|pov|once|yesterday|today i|i remember|we went)\b", re.I),
            'giveaway': re.compile(r"\b(giveaway|win|free|discount|code|sale|drop)\b", re.I),
            'music': re.compile(r"\b(rap|r&b|rnb|playlist|producer|beat|mix|dj|track|song|album|verse|hook)\b", re.I)
        }
    
    def load_and_process_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess Instagram analytics CSV"""
        df = pd.read_csv(csv_path)
        
        # Flexible column mapping (case-insensitive)
        column_mapping = {
            "post id": "post_id",
            "description": "description", 
            "publish time": "publish_time",
            "permalink": "permalink",
            "post type": "post_type",
            "views": "views",
            "reach": "reach", 
            "likes": "likes",
            "shares": "shares",
            "follows": "follows",
            "comments": "comments",
            "saves": "saves",
            "impressions": "impressions"
        }
        
        # Apply column mapping
        lower_cols = {col.lower(): col for col in df.columns}
        for standard_name, mapped_name in column_mapping.items():
            if standard_name in lower_cols:
                actual_col = lower_cols[standard_name]
                if actual_col != mapped_name:
                    df.rename(columns={actual_col: mapped_name}, inplace=True)
        
        # Ensure numeric columns exist
        numeric_cols = ["likes", "comments", "shares", "saves", "follows", "views", "reach", "impressions"]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def calculate_content_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ContentScore with improved follower data handling"""
        df = df.copy()
        
        # Use impressions if available, otherwise fall back to views or reach
        if 'impressions' in df.columns and df['impressions'].sum() > 0:
            impression_col = 'impressions'
        elif 'views' in df.columns and df['views'].sum() > 0:
            impression_col = 'views'
        elif 'reach' in df.columns and df['reach'].sum() > 0:
            impression_col = 'reach'
        else:
            impression_col = None
        
        if impression_col:
            impressions = df[impression_col].fillna(0)
            df['impressions_used'] = impressions
        else:
            impressions = pd.Series(1, index=df.index)
            df['impressions_used'] = 1
        
        # Engagement bundle
        df['engagements'] = df[['likes', 'comments', 'shares', 'saves']].sum(axis=1)
        
        # Find the actual follower column that has data
        follower_columns = ['follows', 'follows_gained', 'followers_gained', 'new_followers', 'followers']
        follower_col = None
        for col in follower_columns:
            if col in df.columns and df[col].notna().any() and (df[col] != 0).any():
                follower_col = col
                break
        
        # Rate calculations with safe division
        if follower_col:
            df['follows_per_impression'] = self._safe_divide(df[follower_col], impressions)
            print(f"Using {follower_col} for follower calculations")
        else:
            df['follows_per_impression'] = 0
            print("Warning: No follower data found for content score calculation")
        
        df['engagement_rate'] = self._safe_divide(df['engagements'], impressions)
        
        # Optional completion rate (if available)
        if 'completion_rate' not in df.columns:
            df['completion_rate'] = 0.5
        
        # Impressions factor: log-scaled and bounded to [0,1]
        log_impressions = np.log1p(impressions)
        max_log_impressions = log_impressions.max() if log_impressions.max() > 0 else 1
        impression_factor = log_impressions / max_log_impressions
        
        # Apply smoothing for low-impression posts
        smoothing_mask = impressions < self.min_impressions
        smoothing_weights = np.where(smoothing_mask, self.smoothing_factor, 1.0)
        
        # Calculate weighted ContentScore
        content_score = (
            self.content_score_weights['follows_per_impression'] * df['follows_per_impression'] * smoothing_weights +
            self.content_score_weights['engagement_rate'] * df['engagement_rate'] * smoothing_weights +
            self.content_score_weights.get('completion_rate', 0.0) * df['completion_rate'] * smoothing_weights
        ) * impression_factor
        
        # Scale to 0-100 and round
        df['content_score'] = (content_score * 100).round(2)
        
        return df
    
    def classify_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced content classification"""
        df = df.copy()
        
        # Hook type classification
        df['hook_type'] = df['description'].apply(self._classify_hook)
        
        # Content type classification  
        df['content_category'] = df.apply(
            lambda row: self._classify_content_type(
                row.get('description', ''), 
                row.get('post_type', '')
            ), axis=1
        )

        # Derived tags per requirements (no NaN in exports)
        df['content_type_tags'] = df.apply(
            lambda row: self._derive_content_type_tags(
                str(row.get('description', '')),
                str(row.get('post_type', ''))
            ), axis=1
        )

        # Per request: keep these columns present but empty for manual filling
        df['audio_type'] = ""
        df['visual_style'] = ""
        df['post_intent'] = ""
        
        df['main_artists'] = df['description'].apply(lambda d: self._extract_main_artist(str(d), getattr(self, 'artist_list_manager', None)))
        df['main_artist'] = df['main_artists']
        df['subgenre'] = df['description'].apply(lambda d: self._infer_subgenre(str(d)))

        return df
    
    def extract_insights(self, df: pd.DataFrame) -> Dict:
        """Extract ranked insights from processed data"""
        insights = {}
        
        # Top performing artists (extract from descriptions)
        artists = self._extract_artists(df, getattr(self, 'artist_list_manager', None))
        if not artists.empty:
            insights['top_artists'] = artists.head(10).to_dict()
        
        # Hook type performance (updated: use engagement bundle metrics)
        hook_performance = df.groupby('hook_type').agg({
            'content_score': ['mean', 'count'],
            'engagement_rate': 'mean',
            'follows_per_impression': 'mean'
        }).round(4)
        hook_performance.columns = ['avg_score', 'post_count', 'avg_engagement_rate', 'avg_follows_per_impression']
        insights['hook_performance'] = hook_performance.sort_values('avg_score', ascending=False).to_dict()
        
        # Optimal posting times
        df['post_hour'] = pd.to_datetime(df.get('publish_time', ''), errors='coerce').dt.hour
        df['time_bucket'] = df['post_hour'].apply(self._get_time_bucket)
        
        time_performance = df.groupby('time_bucket').agg({
            'content_score': ['mean', 'count']
        }).round(4)
        time_performance.columns = ['avg_score', 'post_count']
        insights['optimal_times'] = time_performance.sort_values('avg_score', ascending=False).to_dict()
        
        # Content category insights
        category_performance = df.groupby('content_category').agg({
            'content_score': ['mean', 'count'],
            'engagement_rate': 'mean'
        }).round(4)
        category_performance.columns = ['avg_score', 'post_count', 'avg_engagement']
        insights['category_performance'] = category_performance.sort_values('avg_score', ascending=False).to_dict()
        
        return insights
         
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safe division avoiding division by zero"""
        return np.where((denominator == 0) | pd.isna(denominator), 0, numerator / denominator)
    
    def _classify_hook(self, description: str) -> str:
        """Classify hook type based on description"""
        if not isinstance(description, str) or not description.strip():
            return "Unlabeled"
        
        first_120 = description[:120]
        
        if self.patterns['question'].search(first_120):
            return "Question"
        elif self.patterns['educational'].search(description):
            return "Educational"
        elif self.patterns['trend'].search(description):
            return "Trend Reference"
        elif self.patterns['meme'].search(description):
            return "Meme/Humor"
        elif self.patterns['story'].search(description):
            return "Story/POV"
        elif self.patterns['giveaway'].search(description):
            return "Giveaway/CTA"
        elif self.patterns['music'].search(description):
            return "Music Highlight"
        else:
            return "Unlabeled"
    
    def _classify_content_type(self, description: str, post_type: str) -> str:
        """Classify content type combining post type and content analysis"""
        post_type = (post_type or "").lower()
        
        if "story" in post_type:
            base = "Story"
        elif "reel" in post_type:
            base = "Reel"
        elif "photo" in post_type or "image" in post_type:
            base = "Photo"
        else:
            base = post_type.title() if post_type else "Post"
        
        if not isinstance(description, str):
            description = ""
        
        if self.patterns['educational'].search(description):
            category = "Educational"
        elif self.patterns['story'].search(description):
            category = "Storytelling"
        elif self.patterns['meme'].search(description):
            category = "Meme"
        elif self.patterns['trend'].search(description):
            category = "Trend"
        elif self.patterns['giveaway'].search(description):
            category = "Promo/CTA"
        elif self.patterns['music'].search(description):
            category = "Music Highlight"
        else:
            category = "General"
        
        return f"{base} · {category}"
    
    def _extract_artists(self, df: pd.DataFrame, artist_list_manager=None) -> pd.Series:
        """Extract and rank artists mentioned in descriptions, filtered by uploaded artist list"""
        artist_mentions = {}
        
        # If no artist list manager, return empty series
        if not artist_list_manager:
            return pd.Series(dtype=float)
        
        # Get the name mapping for filtering
        name_mapping = artist_list_manager.get_name_mapping()
        allowed_artists = set(name_mapping.values())  # Get exact names from uploaded list
        
        for desc in df['description'].fillna(''):
            # Extract hashtags and check against uploaded list
            hashtags = re.findall(r'#(\w+)', desc.lower())
            
            for hashtag in hashtags:
                # Try to find a match using the mapping
                normalized_key = re.sub(r"[^a-z0-9]", "", hashtag.lower())
                if normalized_key in name_mapping:
                    artist_name = name_mapping[normalized_key]
                    artist_mentions[artist_name] = artist_mentions.get(artist_name, 0) + 1
                    continue
                
                # Try common variations
                variations = [
                    hashtag.lower().replace(' ', ''),
                    hashtag.lower().replace('$', '').replace('.', '').replace(',', ''),
                    hashtag.lower().replace('the ', ''),
                    hashtag.lower().replace(' ', '_'),
                    # Special case for ASAP -> A$AP
                    hashtag.lower().replace('asap', 'a$ap').replace(' ', ''),
                    hashtag.lower().replace('asap', 'a$ap').replace(' ', '_')
                ]
                
                for var in variations:
                    if var in name_mapping:
                        artist_name = name_mapping[var]
                        artist_mentions[artist_name] = artist_mentions.get(artist_name, 0) + 1
                        break
        
        return pd.Series(artist_mentions).sort_values(ascending=False)

    def _extract_main_artist(self, description: str, artist_list_manager=None) -> str:
        """
        Extract the most likely 'main artist' from a post description.
        Uses hashtags and matches against the uploaded artist list.
        """
        if not description or not isinstance(description, str):
            return ""

        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', description.lower())
        if not hashtags:
            return ""

        # If artist_list_manager is provided, try to match against uploaded list
        if artist_list_manager:
            # Get the name mapping from the artist list
            name_mapping = artist_list_manager.get_name_mapping()
            
            # Try each hashtag to find a match (prefer later hashtags)
            for hashtag in reversed(hashtags):
                # Try to find a match using the mapping
                normalized_key = re.sub(r"[^a-z0-9]", "", hashtag.lower())
                if normalized_key in name_mapping:
                    artist_name = name_mapping[normalized_key]
                    return artist_name
                
                # Try common variations
                variations = [
                    hashtag.lower().replace(' ', ''),
                    hashtag.lower().replace('$', '').replace('.', '').replace(',', ''),
                    hashtag.lower().replace('the ', ''),
                    hashtag.lower().replace(' ', '_'),
                    # Special case for ASAP -> A$AP
                    hashtag.lower().replace('asap', 'a$ap').replace(' ', ''),
                    hashtag.lower().replace('asap', 'a$ap').replace(' ', '_')
                ]
                
                for var in variations:
                    normalized_var = re.sub(r"[^a-z0-9]", "", var)
                    if normalized_var in name_mapping:
                        artist_name = name_mapping[normalized_var]
                        return artist_name
            
        # Fallback: return the last hashtag as-is if no artist list manager
        fallback_artist = hashtags[-1].title()
        return fallback_artist
    
    def _format_artist_name(self, name: str) -> str:
        """Lightweight formatter without any fixed mappings (title-case words)."""
        if not name:
            return ""
        parts = re.split(r"[_\s]+", str(name).strip())
        out = []
        for part in parts:
            if part:
                if part.startswith("$") or "$" in part:
                    out.append(part.upper())
                elif part.isdigit():
                    out.append(part)
                elif part.lower() in ["the", "of", "and", "or", "in", "on", "at", "to", "for", "with", "by"]:
                    out.append(part.lower())
                else:
                    out.append(part.capitalize())
        return " ".join(out)
    
    def _extract_artist_tokens(self, description: str) -> List[str]:
        """
        Extract 1–2 likely artist name tokens to use as lowercase tags.
        Includes @handles and capitalized words but filters common words.
        """
        if not description:
            return []
        tokens: List[str] = []

        # @handles
        for handle in re.findall(r"@([A-Za-z0-9_\.]+)", description):
            tokens.append(handle.lower())

        # Capitalized words (skip common stopwords)
        stopwords = {'the','this','that','when','what','where','who','how','why','and','but'}
        for word in re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", description):
            lw = word.lower()
            if lw not in stopwords:
                tokens.append(lw)

        # Deduplicate, keep first 2
        uniq = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        return uniq[:2]

    def _derive_content_type_tags(self, description: str, post_type: str) -> str:
        """
        Derive tags approximating client's template (e.g., song_discussion, album_discussion; artist tags).
        Produces a semicolon-separated, lowercased tag list.
        """
        tags: List[str] = []
        desc = (description or '').lower()

        # Core discussion tags
        if re.search(r"\b(album|lp|ep|project)\b", desc):
            tags.append('album_discussion')
        if re.search(r"\b(song|track|single)\b", desc):
            tags.append('song_discussion')

        # Include artist name tokens as tags (lowercased)
        artist_tokens = self._extract_artist_tokens(description)
        tags.extend(artist_tokens)

        # Hashtag-derived tokens (prefer the last few)
        hashtags = [h.lower() for h in re.findall(r"#([A-Za-z0-9_\.]+)", description or '')]
        if hashtags:
            ignore = {
                'music','newsong','newsingle','newalbum','new','viral','reels','reel','song','single','album',
                'mixtape','outnow','stream','listen','follow','like','share','rap','hiphop','hip-hop',
                'rnb','randb','rb','trap','drill','lofi','producer','beat','beats','mix','edit'
            }
            for tag in hashtags[-6:]:
                if tag not in ignore and not tag.isdigit():
                    tags.append(tag.replace('.', '').replace('_', ' ').strip())

        # Fallback to post_type if nothing matched
        pt = (post_type or '').lower()
        if not tags:
            if 'reel' in pt:
                tags.append('reel')
            elif 'story' in pt:
                tags.append('story')

        return "; ".join(dict.fromkeys([t for t in tags if t])) or ""

    def _infer_subgenre(self, description: str) -> str:
        """
        Infer subgenre of music from description.
        Matches common subgenre keywords.
        """
        desc = (description or '').lower()
        # Normalize hashtags too
        hashtags = [h.lower() for h in re.findall(r"#([A-Za-z0-9_\.]+)", description or '')]
        tokens = set(desc.replace('#',' ').replace('_',' ').split()) | set(hashtags)
        # Map common aliases to display form
        alias_map = {
            'hiphop': 'Hip-Hop', 'hip-hop': 'Hip-Hop', 'hip': 'Hip-Hop',
            'rnb': 'R&B', 'randb': 'R&B', 'r&b': 'R&B',
            'boombap': 'Boom Bap', 'boom': 'Boom Bap',
        }
        ordered = [
            'trap','drill','lofi','afrobeats','dancehall','hip-hop','hiphop','r&b','rnb',
            'melodic','pop','rock','house','techno','edm','electronic','indie','country','boom','boombap','boom bap'
        ]
        for key in ordered:
            if key in tokens or key in desc:
                if key in alias_map:
                    return alias_map[key]
                if key == 'boom bap':
                    return 'Boom Bap'
                # Title-case words like 'trap' -> 'Trap'
                return key.title()
        return ""
    
    def _get_time_bucket(self, hour: int) -> str:
        """Convert hour to time bucket - same as AnalyticsEngine"""
        if pd.isna(hour):
            return "Unknown"
        
        hour = int(hour)
        if 0 <= hour < 6:
            return "Night (12a-6a)"
        elif 6 <= hour < 12:
            return "Morning (6a-12p)"
        elif 12 <= hour < 18:
            return "Afternoon (12p-6p)"
        else:
            return "Evening (6p-12a)"