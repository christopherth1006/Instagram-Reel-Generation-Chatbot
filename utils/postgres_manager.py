"""
PostgreSQL Manager for Music Curator Assistant
Handles all PostgreSQL database operations for content generation and artist management
Updated to align with existing database schema from process_upload.py
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
import urllib.parse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

class PostgresManager:
    """Manages PostgreSQL database operations for the application"""
    
    def __init__(self, db_engine: Any):
        self.engine = db_engine
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure required tables exist in the database - aligned with process_upload.py schema"""
        try:
            with self.engine.begin() as conn:
                # Create artist_list table (same as process_upload.py)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS artist_list (
                        id SERIAL PRIMARY KEY,
                        artist_name VARCHAR(255) NOT NULL,
                        genre VARCHAR(100),
                        secondary_genre VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create instagram_data table (same as process_upload.py)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS instagram_data (
                        id SERIAL PRIMARY KEY,
                        -- Base columns from both posts and stories
                        post_id VARCHAR(255),
                        account_id VARCHAR(255),
                        account_username VARCHAR(255),
                        account_name VARCHAR(255),
                        description TEXT,
                        duration_sec FLOAT,
                        publish_time TIMESTAMP,
                        permalink TEXT,
                        post_type VARCHAR(100),
                        
                        -- Engagement metrics
                        views INTEGER,
                        reach INTEGER,
                        likes INTEGER,
                        shares INTEGER,
                        follows INTEGER,
                        comments INTEGER,
                        saves INTEGER,
                        profile_visits INTEGER,
                        replies INTEGER,
                        navigation INTEGER,
                        sticker_taps INTEGER,
                        link_clicks INTEGER,
                        source_file VARCHAR(255),
                        
                        -- Calculated metrics (from process_upload.py)
                        engagements INTEGER,
                        engagement_rate_impr FLOAT,
                        save_rate FLOAT,
                        share_rate FLOAT,
                        follow_conversion FLOAT,
                        ai_post_score FLOAT,
                        content_score FLOAT,
                        dow VARCHAR(20),
                        hour_local INTEGER,
                        time_bucket VARCHAR(50),
                        content_type_tags TEXT,
                        hook_type VARCHAR(100),
                        main_artists VARCHAR(255),
                        subgenre VARCHAR(100),
                        
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS followers_metric_data (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL,
                        followers_count INTEGER NOT NULL,
                        source_file VARCHAR(255),
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date)
                    )
                """))

                
                # Create indexes for better performance
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_instagram_data_score ON instagram_data(content_score)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_instagram_data_artist ON instagram_data(main_artists)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_instagram_data_type ON instagram_data(post_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_artist_list_name ON artist_list(artist_name)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_memory_session ON chat_memory(session_id, timestamp)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_followers_metric_data_date ON followers_metric_data(date)"))
                
        except Exception as e:
            print(f"Error ensuring tables exist: {str(e)}")
    
    # Artist List Management Methods (aligned with process_upload.py)
    
    def get_artist_list(self, list_name: str = None) -> Optional[Dict[str, Any]]:
        """Get artist list from database - returns in format expected by artist_list_manager.py"""
        try:
            with self.engine.connect() as conn:
                artists_df = pd.read_sql("SELECT * FROM artist_list", conn)
                
                if artists_df.empty:
                    return None
                
                # Convert to the format expected by artist_list_manager.py
                artists_data = []
                for _, row in artists_df.iterrows():
                    artists_data.append({
                        'name': row['artist_name'],
                        'genre': row.get('genre', 'Unknown'),
                        'secondary_genre': row.get('secondary_genre', ''),
                        'display_name': row['artist_name'],
                        'search_terms': self._generate_search_terms(row['artist_name'])
                    })
                
                return {
                    'list_name': 'active',
                    'data': artists_data,
                    'metadata': {
                        'total_artists': len(artists_data),
                        'updated_at': datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            print(f"Error getting artist list: {str(e)}")
            return None
    
    def get_all_artist_lists(self) -> Dict[str, Any]:
        """Get all artist lists - since we have one table, return the active list"""
        artist_list = self.get_artist_list()
        if artist_list:
            return {'active': artist_list}
        return {}
    
    def delete_artist_list(self, list_name: str) -> bool:
        """Delete artist list - clears the artist_list table"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM artist_list"))
            return True
            
        except Exception as e:
            print(f"Error deleting artist list: {str(e)}")
            return False
    
    def _generate_search_terms(self, artist_name: str) -> List[str]:
        """Generate search terms for artist matching (same as process_upload.py logic)"""
        terms = []
        name = artist_name or ""
        lower = name.lower()
        terms.append(lower)
        
        # Basic variants (same as process_upload.py)
        terms.append(lower.replace(' ', ''))
        terms.append(lower.replace(' ', '_'))
        terms.append(lower.replace('.', ''))
        
        # Normalized form
        import re
        normalized = re.sub(r"[^a-z0-9]", "", lower)
        terms.append(normalized)
        
        # Handle common handle style @names
        terms.append(lower.replace(' ', '').replace('.', '').replace('_', ''))
        
        return list(dict.fromkeys([t for t in terms if t]))
    
    # Content Analytics Methods (using instagram_data table)
    def add_content_analytics(self, content_type: str, content: str, metadata: Dict[str, Any], 
                            performance_score: float = None, theme: str = None, main_artist: str = None) -> bool:
        """Add content analytics data to instagram_data table"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO instagram_data (
                        post_type, description, content_score, main_artists, 
                        post_id, account_username, likes, comments, shares, saves
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """), (
                    content_type,
                    content,
                    performance_score,
                    main_artist,
                    metadata.get('post_id', f"gen_{datetime.now().timestamp()}"),
                    metadata.get('account_username', 'content_generator'),
                    metadata.get('likes', 0),
                    metadata.get('comments', 0),
                    metadata.get('shares', 0),
                    metadata.get('saves', 0)
                ))
            return True
            
        except Exception as e:
            print(f"Error adding content analytics: {str(e)}")
            return False
    
    def get_high_performing_content(self, content_type: str = None, min_score: float = 70, limit: int = 20) -> List[Dict[str, Any]]:
        """Get high-performing content from instagram_data table"""
        try:
            with self.engine.connect() as conn:
                if content_type:
                    query = text("""
                        SELECT * FROM instagram_data 
                        WHERE content_score >= %s AND post_type = %s
                        ORDER BY content_score DESC
                        LIMIT %s
                    """)
                    results = conn.execute(query, (min_score, content_type, limit))
                else:
                    query = text("""
                        SELECT * FROM instagram_data 
                        WHERE content_score >= %s
                        ORDER BY content_score DESC
                        LIMIT %s
                    """)
                    results = conn.execute(query, (min_score, limit))
                
                content_list = []
                for row in results:
                    content_list.append({
                        'content_type': row.post_type,
                        'content': row.description,
                        'metadata': {
                            'content_score': row.content_score,
                            'post_id': row.post_id,
                            'account_username': row.account_username
                        },
                        'performance_score': row.content_score,
                        'theme': getattr(row, 'content_type_tags', ''),
                        'main_artist': row.main_artists
                    })
                
                return content_list
                
        except Exception as e:
            print(f"Error getting high-performing content: {str(e)}")
            return []
    
    def search_similar_content(self, query: str, content_type: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content in instagram_data table"""
        try:
            with self.engine.connect() as conn:
                if content_type:
                    sql = text("""
                        SELECT * FROM instagram_data 
                        WHERE post_type = %s AND (
                            description ILIKE %s OR 
                            content_type_tags ILIKE %s OR
                            main_artists ILIKE %s
                        )
                        ORDER BY content_score DESC
                        LIMIT %s
                    """)
                    results = conn.execute(sql, (content_type, f'%{query}%', f'%{query}%', f'%{query}%', top_k))
                else:
                    sql = text("""
                        SELECT * FROM instagram_data 
                        WHERE description ILIKE %s OR content_type_tags ILIKE %s OR main_artists ILIKE %s
                        ORDER BY content_score DESC
                        LIMIT %s
                    """)
                    results = conn.execute(sql, (f'%{query}%', f'%{query}%', f'%{query}%', top_k))
                
                content_list = []
                for row in results:
                    content_list.append({
                        'content_type': row.post_type,
                        'content': row.description,
                        'metadata': {
                            'content_score': row.content_score,
                            'post_id': row.post_id,
                            'account_username': row.account_username
                        },
                        'performance_score': row.content_score,
                        'theme': getattr(row, 'content_type_tags', ''),
                        'main_artist': row.main_artists
                    })
                
                return content_list
                
        except Exception as e:
            print(f"Error searching similar content: {str(e)}")
            return []
    
    # Artist Performance Methods (using instagram_data table)
    def update_artist_performance(self, artist_name: str, performance_score: float = None, 
                                engagement_rate: float = None, content_count: int = None) -> bool:
        """Update artist performance - not directly stored, calculated on demand"""
        # For this implementation, we calculate performance on the fly from instagram_data
        return True
    
    def get_artist_performance(self, artist_names: List[str] = None, min_score: float = 70) -> List[Dict[str, Any]]:
        """Get artist performance data calculated from instagram_data"""
        try:
            with self.engine.connect() as conn:
                if artist_names:
                    # Convert artist_names to a format suitable for SQL IN clause
                    artists_tuple = tuple(artist_names)
                    if len(artists_tuple) == 1:
                        artists_tuple = f"('{artists_tuple[0]}')"
                    
                    query = text(f"""
                        SELECT 
                            main_artists as artist_name,
                            AVG(content_score) as performance_score,
                            AVG(engagement_rate_impr) as engagement_rate,
                            COUNT(*) as content_count
                        FROM instagram_data 
                        WHERE main_artists IN {artists_tuple} 
                            AND content_score >= %s
                        GROUP BY main_artists
                        ORDER BY performance_score DESC
                    """)
                    results = conn.execute(query, (min_score,))
                else:
                    query = text("""
                        SELECT 
                            main_artists as artist_name,
                            AVG(content_score) as performance_score,
                            AVG(engagement_rate_impr) as engagement_rate,
                            COUNT(*) as content_count
                        FROM instagram_data 
                        WHERE content_score >= %s
                            AND main_artists IS NOT NULL 
                            AND main_artists != ''
                        GROUP BY main_artists
                        ORDER BY performance_score DESC
                        LIMIT 50
                    """)
                    results = conn.execute(query, (min_score,))
                
                performance_data = []
                for row in results:
                    performance_data.append({
                        'artist_name': row.artist_name,
                        'performance_score': float(row.performance_score) if row.performance_score else 0,
                        'engagement_rate': float(row.engagement_rate) if row.engagement_rate else 0,
                        'content_count': row.content_count
                    })
                
                return performance_data
                
        except Exception as e:
            print(f"Error getting artist performance: {str(e)}")
            return []
    
    # Analytics and Insights Methods
    def get_performance_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get overall performance insights from instagram_data"""
        try:
            with self.engine.connect() as conn:
                insights = {}
                
                # Get average performance score
                query = text("""
                    SELECT 
                        AVG(content_score) as avg_score,
                        COUNT(*) as total_content,
                        COUNT(DISTINCT main_artists) as unique_artists
                    FROM instagram_data 
                    WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
                """)
                result = conn.execute(query, (days,)).fetchone()
                
                insights['avg_performance_score'] = float(result.avg_score) if result.avg_score else 0
                insights['total_content'] = result.total_content
                insights['unique_artists'] = result.unique_artists
                
                # Get top performing artists
                query = text("""
                    SELECT 
                        main_artists,
                        AVG(content_score) as avg_score,
                        COUNT(*) as content_count
                    FROM instagram_data 
                    WHERE main_artists IS NOT NULL 
                        AND main_artists != ''
                        AND created_at >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY main_artists
                    HAVING COUNT(*) >= 1
                    ORDER BY avg_score DESC
                    LIMIT 10
                """)
                results = conn.execute(query, (days,))
                
                top_artists = {}
                for row in results:
                    top_artists[row.main_artists] = {
                        'avg_score': float(row.avg_score),
                        'content_count': row.content_count
                    }
                insights['top_artists'] = top_artists
                
                # Get common themes from content_type_tags
                query = text("""
                    SELECT 
                        content_type_tags,
                        COUNT(*) as count,
                        AVG(content_score) as avg_score
                    FROM instagram_data 
                    WHERE content_type_tags IS NOT NULL 
                        AND content_type_tags != ''
                        AND created_at >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY content_type_tags
                    ORDER BY count DESC, avg_score DESC
                    LIMIT 5
                """)
                results = conn.execute(query, (days,))
                
                themes = []
                for row in results:
                    themes.append({
                        'theme': row.content_type_tags,
                        'count': row.count,
                        'avg_score': float(row.avg_score)
                    })
                insights['common_themes'] = themes
                
                return insights
                
        except Exception as e:
            print(f"Error getting performance insights: {str(e)}")
            return {}
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a custom SQL query and return results"""
        try:
            with self.engine.connect() as conn:
                if params:
                    results = conn.execute(text(query), params)
                else:
                    results = conn.execute(text(query))
                
                return [dict(row) for row in results]
                
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return []

    def _ensure_chat_memory_table(self):
        """Ensure chat_memory table exists"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_memory (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create index for better performance
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_chat_memory_session 
                    ON chat_memory(session_id, timestamp)
                """))
        except Exception as e:
            print(f"Error ensuring chat_memory table: {e}")        

    def get_followers_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get followers data from database"""
        try:
            with self.engine.connect() as conn:
                query = "SELECT date, followers_count, source_file FROM followers_metric_data"
                params = {}
                
                if start_date or end_date:
                    query += " WHERE 1=1"
                    if start_date:
                        query += " AND date >= :start_date"
                        params['start_date'] = start_date
                    if end_date:
                        query += " AND date <= :end_date"
                        params['end_date'] = end_date
                
                query += " ORDER BY date"
                
                df = pd.read_sql(text(query), conn, params=params)
                return df
                
        except Exception as e:
            print(f"Error getting followers data: {str(e)}")
            return pd.DataFrame()

    def get_followers_growth_metrics(self, period_days: int = 30) -> Dict[str, Any]:
        """Calculate followers growth metrics"""
        try:
            with self.engine.connect() as conn:
                # Get the most recent date and calculate period start
                result = conn.execute(text("""
                    SELECT MAX(date) as latest_date FROM followers_metric_data
                """))
                latest_date = result.scalar()
                
                if not latest_date:
                    return {}
                
                period_start = latest_date - pd.Timedelta(days=period_days)
                
                # Calculate growth metrics
                result = conn.execute(text("""
                    WITH period_data AS (
                        SELECT 
                            MIN(followers_count) as start_followers,
                            MAX(followers_count) as end_followers,
                            COUNT(*) as days_count
                        FROM followers_metric_data 
                        WHERE date BETWEEN :period_start AND :latest_date
                    )
                    SELECT 
                        start_followers,
                        end_followers,
                        end_followers - start_followers as total_growth,
                        (end_followers - start_followers) * 100.0 / NULLIF(start_followers, 0) as growth_percentage,
                        days_count
                    FROM period_data
                """), {
                    'period_start': period_start,
                    'latest_date': latest_date
                })
                
                metrics = result.fetchone()
                
                if metrics:
                    return {
                        'period_start': period_start,
                        'period_end': latest_date,
                        'start_followers': metrics[0] or 0,
                        'end_followers': metrics[1] or 0,
                        'total_growth': metrics[2] or 0,
                        'growth_percentage': metrics[3] or 0,
                        'days_count': metrics[4] or 0,
                        'avg_daily_growth': (metrics[2] or 0) / max(metrics[4] or 1, 1)
                    }
                return {}
                
        except Exception as e:
            print(f"Error getting followers growth metrics: {str(e)}")
            return {}