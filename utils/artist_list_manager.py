"""
Artist List Manager for Music Curator Assistant
Handles artist list uploads, management, and integration with content generation
"""
import pandas as pd
import streamlit as st
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

class ArtistListManager:
    """Manages artist lists using PostgreSQL"""
    
    def __init__(self, db_engine=None):  # Changed from postgres_manager to db_engine
        self.db_engine = db_engine  # Use db_engine directly to match your pattern
        # Keep in-memory cache for fast matching
        self._normalized_index_cache: Dict[str, Dict[str, str]] = {}
        
    def set_db_engine(self, db_engine):
        """Set the db engine (injected from main app)"""
        self.db_engine = db_engine
    
    def process_artist_list_upload(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded artist list CSV or XLSX file"""
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Please use CSV or XLSX.")
            
            # Validate required columns
            required_columns = ['Artist Name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Clean and process the data
            processed_data = self._process_artist_dataframe(df)
            
            # Create metadata
            metadata = {
                'filename': uploaded_file.name,
                'upload_time': datetime.now().isoformat(),
                'total_artists': len(processed_data),
                'genres': list(set([artist.get('genre', 'Unknown') for artist in processed_data])),
                'secondary_genres': list(set([artist.get('secondary_genre', '') for artist in processed_data if artist.get('secondary_genre')]))
            }
            
            return {
                'data': processed_data,
                'metadata': metadata
            }
            
        except Exception as e:
            raise Exception(f"Error processing artist list: {str(e)}")
    
    def _process_artist_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process and clean artist dataframe"""
        processed_artists = []
        
        for _, row in df.iterrows():
            artist_name = str(row.get('Artist Name', '')).strip()
            if not artist_name or artist_name.lower() in ['nan', '']:
                continue
                
            # Preserve artist name exactly as uploaded
            artist_name = self._clean_artist_name(artist_name)
            
            # Extract genre information
            genre = str(row.get('Genre', '')).strip() if 'Genre' in row else 'Unknown'
            secondary_genre = str(row.get('Secondary Genre', '')).strip() if 'Secondary Genre' in row else ''
            
            # Create artist entry
            artist_entry = {
                'name': artist_name,
                'genre': genre,
                'secondary_genre': secondary_genre if secondary_genre and secondary_genre.lower() not in ['nan', ''] else '',
                'display_name': artist_name,
                'search_terms': self._generate_search_terms(artist_name)
            }
            
            processed_artists.append(artist_entry)
        
        return processed_artists
    
    def _clean_artist_name(self, name: str) -> str:
        """Trim whitespace; preserve uploader's original casing and symbols."""
        return ' '.join(str(name).split())

    def _normalize(self, text: str) -> str:
        """Normalization used for matching (case-insensitive, remove non-alnum)."""
        import re
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())
    
    def _generate_search_terms(self, artist_name: str) -> List[str]:
        """Generate search terms for artist matching"""
        terms: List[str] = []
        name = artist_name or ""
        lower = name.lower()
        terms.append(lower)
        # Basic variants
        terms.append(lower.replace(' ', ''))
        terms.append(lower.replace(' ', '_'))
        terms.append(lower.replace('.', ''))
        # Normalized form
        terms.append(self._normalize(name))
        # Handle common handle style @names
        terms.append(lower.replace(' ', '').replace('.', '').replace('_', ''))
        return list(dict.fromkeys([t for t in terms if t]))
    
    def add_artist_list(self, list_name: str, artist_data: Dict[str, Any]) -> bool:
        """Add a new artist list to PostgreSQL - Direct database insertion"""
        if not self.db_engine:
            return False
        
        try:
            # Direct database insertion instead of using postgres_manager methods
            artists_df = pd.DataFrame(artist_data['data'])
            
            # Rename columns to match database schema
            column_mapping = {
                'name': 'artist_name',
                'genre': 'genre', 
                'secondary_genre': 'secondary_genre'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in artists_df.columns:
                    artists_df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns exist
            if 'artist_name' not in artists_df.columns:
                return False
            
            # Upload to artist_list table
            artists_df.to_sql(
                "artist_list",
                con=self.db_engine,
                if_exists="append",
                index=False,
                method="multi"
            )
            
            # Update local cache
            self._attach_normalized_index(list_name, artist_data)
            return True
            
        except Exception as e:
            print(f"Error adding artist list to database: {str(e)}")
            return False
    
    def update_artist_list(self, list_name: str, artist_data: Dict[str, Any]) -> bool:
        """Update an existing artist list in PostgreSQL"""
        if not self.db_engine:
            return False
        
        try:
            # Delete existing list and re-add
            self.delete_artist_list(list_name)
            success = self.add_artist_list(list_name, artist_data)
            if success:
                # Update local cache
                self._attach_normalized_index(list_name, artist_data)
            return success
            
        except Exception as e:
            print(f"Error updating artist list in database: {str(e)}")
            return False
    
    def delete_artist_list(self, list_name: str) -> bool:
        """Delete an artist list from PostgreSQL"""
        if not self.db_engine:
            return False
        
        try:
            from sqlalchemy import text
            
            with self.db_engine.begin() as conn:
                conn.execute(text("DELETE FROM artist_list"))
                
            if list_name in self._normalized_index_cache:
                del self._normalized_index_cache[list_name]
            return True
            
        except Exception as e:
            print(f"Error deleting artist list from database: {str(e)}")
            return False
    
    def get_artist_list(self, list_name: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific artist list from PostgreSQL"""
        if not self.db_engine:
            return None
        
        try:
            from sqlalchemy import text
            
            with self.db_engine.connect() as conn:
                artists_df = pd.read_sql("SELECT * FROM artist_list", conn)
                
            if artists_df.empty:
                return None
            
            # Convert to the format expected by the rest of the application
            artists_data = []
            for _, row in artists_df.iterrows():
                artists_data.append({
                    'name': row['artist_name'],
                    'genre': row.get('genre', 'Unknown'),
                    'secondary_genre': row.get('secondary_genre', ''),
                    'display_name': row['artist_name'],
                    'search_terms': self._generate_search_terms(row['artist_name'])
                })
            
            artist_list_data = {
                'list_name': 'active',
                'data': artists_data,
                'metadata': {
                    'total_artists': len(artists_data),
                    'updated_at': datetime.now().isoformat()
                }
            }
            
            # Update local cache
            self._attach_normalized_index(list_name or 'active', artist_list_data)
            return artist_list_data
            
        except Exception as e:
            print(f"Error getting artist list from database: {str(e)}")
            return None
    
    def get_all_artist_lists(self) -> Dict[str, Any]:
        """Get all artist lists from PostgreSQL - since we have one table, return the active list"""
        artist_list = self.get_artist_list()
        if artist_list:
            return {'active': artist_list}
        return {}
    
    def get_active_artist_list(self) -> Optional[Dict[str, Any]]:
        """Get the currently active artist list from PostgreSQL"""
        return self.get_artist_list()
    
    def set_active_artist_list(self, list_name: str) -> bool:
        """Set the active artist list"""
        # This is a simple implementation - we only have one list
        return True
    
    def get_artist_names_for_prompt(self, list_name: Optional[str] = None) -> List[str]:
        """Get artist names formatted for use in prompts"""
        artist_list = self.get_artist_list(list_name)
        
        if not artist_list:
            return []
        
        artists_data = artist_list.get('data') or []
        return [artist['name'] for artist in artists_data]
    
    def get_artist_genres_for_prompt(self, list_name: Optional[str] = None) -> Dict[str, List[str]]:
        """Get artists grouped by genre for use in prompts"""
        artist_list = self.get_artist_list(list_name)
        
        if not artist_list:
            return {}
        
        artists_data = artist_list.get('data') or []
        genres = {}
        for artist in artists_data:
            genre = artist.get('genre', 'Unknown')
            if genre not in genres:
                genres[genre] = []
            genres[genre].append(artist['name'])
        
        return genres
    
    def find_artist_match(self, artist_name: str, list_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find if an artist matches any in the active list"""
        artist_list = self.get_artist_list(list_name)
        if not artist_list or 'data' not in artist_list:
            return None
        
        normalized_index = self._ensure_index(list_name, artist_list)
        cand = self._normalize(artist_name)
        if cand in normalized_index:
            exact = normalized_index[cand]
            # return the full artist dict for the exact name
            for a in artist_list['data']:
                if a['name'] == exact:
                    return a
        # try relaxed: remove trailing 'the'
        if cand.startswith('the'):
            cand2 = cand[3:]
            if cand2 in normalized_index:
                exact = normalized_index[cand2]
                for a in artist_list['data']:
                    if a['name'] == exact:
                        return a
        return None

    def _attach_normalized_index(self, list_name: str, artist_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create normalized->exact index and attach to the list payload; also cache in memory."""
        idx: Dict[str, str] = {}
        for a in artist_data.get('data', []) or []:
            exact = a.get('name', '')
            if not exact:
                continue
            variants = set(a.get('search_terms', []) or [])
            variants.add(self._normalize(exact))
            variants.add(exact.lower())
            variants.add(exact.lower().replace(' ', ''))
            variants.add(exact.lower().replace('.', ''))
            variants.add(exact.lower().replace('_', ''))
            for v in variants:
                idx[self._normalize(v)] = exact
        artist_data['__normalized_index__'] = idx
        self._normalized_index_cache[list_name or 'active'] = idx
        return artist_data

    def _ensure_index(self, list_name: Optional[str], artist_list: Dict[str, Any]) -> Dict[str, str]:
        key = list_name or 'active'
        if key in self._normalized_index_cache:
            return self._normalized_index_cache[key]
        idx = artist_list.get('__normalized_index__') or {}
        if not idx:
            # build on the fly
            rebuilt = self._attach_normalized_index(key, artist_list)
            idx = rebuilt.get('__normalized_index__') or {}
        self._normalized_index_cache[key] = idx
        return idx
    
    def get_name_mapping(self) -> Dict[str, str]:
        """Create a normalized name mapping from the active artist list for matching."""
        active_list = self.get_active_artist_list()
        if not active_list:
            return {}
        
        artists_data = active_list.get('data', [])
        if not artists_data:
            return {}
        
        mapping = {}
        for artist_entry in artists_data:
            name = artist_entry['name']
            # Create normalized key (lowercase, no spaces, no special chars)
            normalized_key = re.sub(r"[^a-z0-9]", "", name.lower())
            mapping[normalized_key] = name
            
            # Add common variations
            variations = [
                name.lower().replace(' ', ''),
                name.lower().replace('$', '').replace('.', '').replace(',', ''),
                name.lower().replace('the ', ''),
                name.lower().replace(' ', '_'),
                # Special case for A$AP -> ASAP
                name.lower().replace('a$ap', 'asap').replace(' ', ''),
                name.lower().replace('a$ap', 'asap').replace(' ', '_')
            ]
            for var in variations:
                if var not in mapping:
                    mapping[var] = name
        
        return mapping

    def get_artist_filter_prompt(self, list_name: Optional[str] = None) -> str:
        """Generate strict artist filter prompt for content generation"""
        artist_list = self.get_artist_list(list_name)
        
        if not artist_list:
            return ""
        
        artists_data = artist_list.get('data') or []
        artist_names = [artist['name'] for artist in artists_data]
        genres = self.get_artist_genres_for_prompt(list_name)
        
        prompt_parts = []
        
        # Add strict artist restriction
        if artist_names:
            prompt_parts.append(f"RESTRICTION: You are FORBIDDEN from using any artists not in this exact list: {', '.join(artist_names)}")
            prompt_parts.append(f"ONLY use these {len(artist_names)} artists: {', '.join(artist_names[:15])}")
            if len(artist_names) > 15:
                prompt_parts.append(f"Plus these additional artists: {', '.join(artist_names[15:30])}")
                if len(artist_names) > 30:
                    prompt_parts.append(f"And {len(artist_names) - 30} more from the uploaded list")
        
        # Add genre information for context
        if genres:
            genre_info = []
            for genre, artists in genres.items():
                if len(artists) <= 5:
                    genre_info.append(f"{genre}: {', '.join(artists)}")
                else:
                    genre_info.append(f"{genre}: {', '.join(artists[:3])} and {len(artists) - 3} more")
            
            if genre_info:
                prompt_parts.append(f"Available genres from your list: {'; '.join(genre_info)}")
        
        return " ".join(prompt_parts)
    
    def validate_artist_list_file(self, uploaded_file) -> Dict[str, Any]:
        """Validate uploaded CSV or XLSX file before processing"""
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, nrows=5)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, nrows=5)
            else:
                return {
                    'valid': False,
                    'errors': [f"Unsupported file format: {file_extension}. Please use CSV or XLSX."],
                    'warnings': [],
                    'columns': [],
                    'sample_data': []
                }
            
            validation_result = {
                'valid': True,
                'columns': list(df.columns),
                'sample_data': df.head(3).to_dict('records'),
                'warnings': [],
                'errors': []
            }
            
            # Check for required columns
            required_columns = ['Artist Name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Check for optional columns
            optional_columns = ['Genre', 'Secondary Genre']
            missing_optional = [col for col in optional_columns if col not in df.columns]
            
            if missing_optional:
                validation_result['warnings'].append(f"Optional columns not found: {', '.join(missing_optional)}")
            
            # Check for empty data
            if df.empty:
                validation_result['valid'] = False
                validation_result['errors'].append("File appears to be empty")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Error reading file: {str(e)}"],
                'warnings': [],
                'columns': [],
                'sample_data': []
            }