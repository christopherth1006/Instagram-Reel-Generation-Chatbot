import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
import urllib.parse
from datetime import datetime
import re
import io
import time
from sqlalchemy.engine import Engine

@st.cache_resource
def get_engine():
    """Cached PostgreSQL connection engine."""
    db = st.secrets["postgres"]
    password = urllib.parse.quote_plus(db["password"])
    conn_str = f"postgresql+psycopg2://{db['user']}:{password}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(conn_str, pool_pre_ping=True)

def get_artist_list(engine):
    """Get the current artist list from database"""
    try:
        with engine.connect() as conn:
            # Check if artist_list table exists
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'artist_list')"
            ))
            table_exists = result.scalar()
            
            if table_exists:
                artists_df = pd.read_sql("SELECT * FROM artist_list", conn)
                return artists_df
            else:
                return pd.DataFrame()
    except Exception as e:
        print(f"Error getting artist list: {e}")
        return pd.DataFrame()

def create_artist_name_mapping(artists_df):
    """Create a mapping from normalized names to actual artist names"""
    if artists_df.empty:
        return {}
    
    mapping = {}
    
    for _, artist in artists_df.iterrows():
        artist_name = artist.get('artist_name', '')
        if not artist_name or pd.isna(artist_name):
            continue
            
        # Add the exact name
        exact_name = str(artist_name).strip()
        normalized_key = re.sub(r"[^a-z0-9]", "", exact_name.lower())
        mapping[normalized_key] = exact_name
        
        # Add common variations
        variations = [
            exact_name.lower().replace(' ', ''),
            exact_name.lower().replace('$', '').replace('.', '').replace(',', ''),
            exact_name.lower().replace('the ', ''),
            exact_name.lower().replace(' ', '_'),
            # Special case for A$AP -> ASAP
            exact_name.lower().replace('a$ap', 'asap').replace(' ', ''),
            exact_name.lower().replace('a$ap', 'asap').replace(' ', '_'),
            # Remove common suffixes
            exact_name.lower().replace('official', '').replace('music', '').replace('vevo', '').strip()
        ]
        
        for var in variations:
            if var:
                normalized_var = re.sub(r"[^a-z0-9]", "", var)
                if normalized_var and normalized_var not in mapping:
                    mapping[normalized_var] = exact_name
    
    return mapping

def extract_main_artist_from_description(description, artist_mapping):
    """Extract main artist from description using artist list mapping"""
    if not isinstance(description, str) or not description.strip():
        return ""
    
    # Extract hashtags
    hashtags = re.findall(r'#(\w+)', description.lower())
    if not hashtags:
        return ""
    
    # Try each hashtag to find a match in artist list (prefer later hashtags)
    for hashtag in reversed(hashtags):
        # Normalize the hashtag
        normalized_hashtag = re.sub(r"[^a-z0-9]", "", hashtag.lower())
        
        # Check for exact match
        if normalized_hashtag in artist_mapping:
            return artist_mapping[normalized_hashtag]
        
        # Try common variations
        variations = [
            hashtag.lower().replace(' ', ''),
            hashtag.lower().replace('$', '').replace('.', '').replace(',', ''),
            hashtag.lower().replace('the ', ''),
            hashtag.lower().replace(' ', '_'),
            hashtag.lower().replace('a$ap', 'asap').replace(' ', ''),
            hashtag.lower().replace('a$ap', 'asap').replace(' ', '_'),
            hashtag.lower().replace('official', '').replace('music', '').replace('vevo', '').strip()
        ]
        
        for var in variations:
            normalized_var = re.sub(r"[^a-z0-9]", "", var)
            if normalized_var in artist_mapping:
                return artist_mapping[normalized_var]
    
    # If no match found, return empty string (don't use artists not in list)
    return ""

def derive_content_type_tags(description: str, post_type: str) -> str:
    """
    Derive content_type_tags similar to analytics_engine.py
    Produces a semicolon-separated, lowercased tag list.
    """
    tags = []
    desc = (description or '').lower()
    post_type = (post_type or '').lower()

    # Core discussion tags
    if re.search(r"\b(album|lp|ep|project)\b", desc):
        tags.append('album_discussion')
    if re.search(r"\b(song|track|single)\b", desc):
        tags.append('song_discussion')

    # Extract artist tokens for tags
    artist_tokens = extract_artist_tokens_for_tags(description)
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
    if not tags:
        if 'reel' in post_type:
            tags.append('reel')
        elif 'story' in post_type:
            tags.append('story')
        elif 'photo' in post_type or 'image' in post_type:
            tags.append('photo')
        else:
            tags.append('post')

    return "; ".join(dict.fromkeys([t for t in tags if t])) or ""

def extract_artist_tokens_for_tags(description: str):
    """
    Extract 1–2 likely artist name tokens to use as lowercase tags.
    Includes @handles and capitalized words but filters common words.
    """
    if not description:
        return []
    tokens = []

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

def calculate_metrics(df, file_type, engine):
    """Calculate all necessary metrics for Instagram data using analytics_engine logic"""
    df = df.copy()
    
    # Get artist list for validation
    artists_df = get_artist_list(engine)
    artist_mapping = create_artist_name_mapping(artists_df)
    
    # Determine impression source (same as analytics_engine.py)
    if 'impressions' in df.columns and df['impressions'].notna().any():
        impression_col = 'impressions'
    elif 'views' in df.columns and df['views'].notna().any():
        impression_col = 'views'
    elif 'reach' in df.columns and df['reach'].notna().any():
        impression_col = 'reach'
    else:
        impression_col = None
    
    # Safe impression value for calculations
    if impression_col:
        impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
        df['impressions_used'] = impressions
    else:
        impressions = pd.Series(1, index=df.index)
        df['impressions_used'] = 1
    
    # Engagement bundle (same as analytics_engine.py)
    df['engagements'] = df[['likes', 'comments', 'shares', 'saves']].sum(axis=1, skipna=True)
    
    # Find the actual follower column that has data
    follower_columns = ['follows', 'follows_gained', 'followers_gained', 'new_followers', 'followers']
    follower_col = None
    for col in follower_columns:
        if col in df.columns and df[col].notna().any() and (df[col] != 0).any():
            follower_col = col
            break
    
    # Rate calculations with safe division (same as analytics_engine.py)
    if follower_col:
        df['follows_per_impression'] = _safe_divide(df[follower_col], impressions)
    else:
        df['follows_per_impression'] = 0
    
    df['engagement_rate'] = _safe_divide(df['engagements'], impressions)
    
    # Optional completion rate (if available)
    if 'completion_rate' not in df.columns:
        df['completion_rate'] = 0.5
    
    # Content Score Calculation (EXACTLY like analytics_engine.py)
    # Use the same weights and parameters
    content_score_weights = {
        'follows_per_impression': 0.5,
        'engagement_rate': 0.5
    }
    min_impressions_threshold = 100
    smoothing_factor = 0.5
    
    # Impressions factor: log-scaled and bounded to [0,1]
    log_impressions = np.log1p(impressions)
    max_log_impressions = log_impressions.max() if log_impressions.max() > 0 else 1
    impression_factor = log_impressions / max_log_impressions
    
    # Apply smoothing for low-impression posts
    smoothing_mask = impressions < min_impressions_threshold
    smoothing_weights = np.where(smoothing_mask, smoothing_factor, 1.0)
    
    # Calculate weighted ContentScore (EXACTLY like analytics_engine.py)
    content_score = (
        content_score_weights['follows_per_impression'] * df['follows_per_impression'] * smoothing_weights +
        content_score_weights['engagement_rate'] * df['engagement_rate'] * smoothing_weights +
        content_score_weights.get('completion_rate', 0.0) * df['completion_rate'] * smoothing_weights
    ) * impression_factor
    
    # Scale to 0-100 and round (EXACTLY like analytics_engine.py)
    df['content_score'] = (content_score * 100).round(2)
    df['ai_post_score'] = df['content_score']  # Keep both for compatibility
    
    # Convert rates to percentages for display (like analytics_engine.py)
    df['engagement_rate_impr'] = (df['engagement_rate'] * 100).round(2)
    df['follow_conversion'] = (df['follows_per_impression'] * 100).round(2)
    
    # Calculate additional rates for completeness
    df['save_rate'] = _safe_divide(df.get('saves', 0), impressions) * 100
    df['share_rate'] = _safe_divide(df.get('shares', 0), impressions) * 100
    
    # Time-based metrics
    if 'publish_time' in df.columns:
        publish_dt = pd.to_datetime(df['publish_time'], errors='coerce')
        df['dow'] = publish_dt.dt.day_name().fillna('Unknown')
        df['hour_local'] = publish_dt.dt.hour.fillna(0)
        df['time_bucket'] = df['hour_local'].apply(get_time_bucket)
    else:
        df['dow'] = 'Unknown'
        df['hour_local'] = 0
        df['time_bucket'] = 'Unknown'
    
    # Content classification with artist validation
    df = classify_content(df, artist_mapping)
    
    # Ensure all required calculated columns exist with proper defaults
    calculated_columns = {
        'content_type_tags': '',
        'hook_type': '',
        'main_artists': '',
        'subgenre': ''
    }
    
    for col, default in calculated_columns.items():
        if col not in df.columns:
            df[col] = default
    
    # Format percentage columns to 2 decimal places
    percent_cols = ['engagement_rate_impr', 'save_rate', 'share_rate', 'follow_conversion']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    
    # Ensure numeric fields are properly typed
    numeric_fields = ['engagements', 'engagement_rate_impr', 'save_rate', 'share_rate', 
                     'follow_conversion', 'ai_post_score', 'content_score', 'hour_local']
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    return df

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division avoiding division by zero (same as analytics_engine.py)"""
    return np.where((denominator == 0) | pd.isna(denominator), 0, numerator / denominator)

def classify_content(df, artist_mapping):
    """Content classification using same logic as analytics_engine.py"""
    df = df.copy()
    
    # Pattern matching for hook types (same as analytics_engine.py)
    patterns = {
        'question': re.compile(r"\?\s*$|^\s*(how|what|why|who|when|where|is|are|can|should|do|does|did)\b", re.I),
        'educational': re.compile(r"\b(how to|tutorial|tips|guide|secrets|mistakes|thread|breakdown|explained)\b", re.I),
        'trend': re.compile(r"\b(trend|challenge|viral|capcut|template)\b", re.I),
        'meme': re.compile(r"\b(meme|lol|funny|joke|humor|lmao)\b", re.I),
        'story': re.compile(r"\b(story|pov|once|yesterday|today i|i remember|we went)\b", re.I),
        'giveaway': re.compile(r"\b(giveaway|win|free|discount|code|sale|drop)\b", re.I),
        'music': re.compile(r"\b(rap|r&b|rnb|playlist|producer|beat|mix|dj|track|song|album|verse|hook)\b", re.I)
    }
    
    def classify_hook(description):
        if not isinstance(description, str) or not description.strip():
            return "Unlabeled"
        
        first_120 = description[:120]
        
        for hook_type, pattern in patterns.items():
            if pattern.search(first_120):
                return hook_type.title()
        
        return "Unlabeled"
    
    if 'description' in df.columns:
        df['hook_type'] = df['description'].apply(classify_hook)
        
        # Extract main artist from hashtags using artist list validation
        df['main_artists'] = df['description'].apply(
            lambda desc: extract_main_artist_from_description(desc, artist_mapping)
        )
        
        # Calculate content_type_tags using analytics_engine logic
        df['content_type_tags'] = df.apply(
            lambda row: derive_content_type_tags(
                str(row.get('description', '')),
                str(row.get('post_type', ''))
            ), axis=1
        )
        
        # Simple subgenre detection (same as analytics_engine.py)
        def detect_subgenre(desc):
            if not isinstance(desc, str):
                return ""
            desc_lower = desc.lower()
            if any(word in desc_lower for word in ['rap', 'hiphop', 'hip-hop']):
                return "Hip-Hop"
            elif any(word in desc_lower for word in ['r&b', 'rnb', 'soul']):
                return "R&B"
            elif any(word in desc_lower for word in ['pop', 'popular']):
                return "Pop"
            elif any(word in desc_lower for word in ['rock', 'alternative']):
                return "Rock"
            elif any(word in desc_lower for word in ['electronic', 'edm', 'house']):
                return "Electronic"
            return ""
        
        df['subgenre'] = df['description'].apply(detect_subgenre)
    
    return df

def get_time_bucket(hour):
    """Convert hour to time bucket (same as analytics_engine.py)"""
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

def process_uploaded_files(uploaded_files):
    """Upload posts, stories, or artist lists to respective tables"""
    engine = get_engine()

    # Define all columns for the unified instagram_data table
    instagram_columns = [
        # Base columns from both posts and stories
        "post_id", "account_id", "account_username", "account_name", "description",
        "duration_(sec)", "publish_time", "permalink", "post_type",
        "views", "reach", "likes", "shares", "follows", "comments", "saves",
        "profile_visits", "replies", "navigation", "sticker_taps", "link_clicks", "source_file",
        
        # Calculated metrics (now using analytics_engine logic)
        "engagements", "engagement_rate_impr", "save_rate", "share_rate", "follow_conversion",
        "ai_post_score", "content_score", "dow", "hour_local", "time_bucket", "content_type_tags", "hook_type",
        "main_artists", "subgenre"
    ]

    # Artist list columns
    artist_columns = ["artist_name", "genre", "secondary_genre"]

    print(f"uploaded files: {uploaded_files}")
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    for file in uploaded_files:
        try:
            # Get file name safely
            if hasattr(file, "name"):
                fname = file.name
            else:
                fname = getattr(file, "filename", "uploaded_file")
            st.info(f"📄 Processing: {fname}")
            
            # Read file
            if fname.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file)
            
            # Standardize column names
            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
            fname_lower = fname.lower()

            # Handle artist list uploads separately
            if "artist" in fname_lower or "artists" in fname_lower:
                # Process artist list
                process_artist_list(df, fname, engine)
                continue
            
            # Fill missing base columns
            base_columns = [
                "post_id", "account_id", "account_username", "account_name", "description",
                "duration_(sec)", "publish_time", "permalink", "post_type",
                "views", "reach", "likes", "shares", "follows", "comments", "saves",
                "profile_visits", "replies", "navigation", "sticker_taps", "link_clicks"
            ]

            numeric_cols = [
                "views", "reach", "likes", "shares", "follows", "comments", "saves",
                "profile_visits", "replies", "navigation", "sticker_taps", "link_clicks"
            ]
            
            for col in base_columns:
                if col not in df.columns:
                    # Create the column if missing
                    df[col] = 0 if col in numeric_cols else ""
                else:
                    # Fill empty numeric cells with 0
                    if col in numeric_cols:
                        df[col] = df[col].replace("", 0).fillna(0)

            # Add source_file (required)
            df["source_file"] = fname
            df["processed_at"] = datetime.now()

            # Convert dates
            for dcol in ["publish_time"]:
                if dcol in df.columns:
                    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

            # Calculate all metrics with artist validation USING ANALYTICS_ENGINE LOGIC
            df = calculate_metrics(df, "instagram", engine)

            # Remove duplicates within file
            before = len(df)
            df.drop_duplicates(subset=["post_id"], inplace=True)

            # Remove duplicates already in DB
            try:
                with engine.begin() as conn:
                    # Check if instagram_data table exists
                    result = conn.execute(text(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'instagram_data')"
                    ))
                    table_exists = result.scalar()
                    
                    if table_exists:
                        existing_ids = pd.read_sql(text("SELECT post_id FROM instagram_data"), conn)
                        df = df[~df['post_id'].astype(str).isin(existing_ids['post_id'].astype(str))]
            except SQLAlchemyError:
                # Table doesn't exist yet, will be created
                pass

            after_final = len(df)

            # Upload to single instagram_data table
            if after_final > 0:
                try:
                    # Ensure all calculated columns are present
                    for col in instagram_columns:
                        if col not in df.columns:
                            if col in ['engagements', 'engagement_rate_impr', 'save_rate', 'share_rate', 
                                      'follow_conversion', 'ai_post_score', 'content_score', 'hour_local']:
                                df[col] = 0
                            else:
                                df[col] = ""
                    
                    # Reorder columns to match desired structure
                    df = df[instagram_columns]
                    
                    df.to_sql(
                        "instagram_data",
                        con=engine,
                        if_exists="append",
                        index=False,
                        method="multi"
                    )
                    
                    # Show success message with metrics summary
                    avg_score = df['content_score'].mean()
                    avg_engagement = df['engagement_rate_impr'].mean()
                    matched_artists = df['main_artists'].str.len() > 0
                    artist_match_rate = matched_artists.sum() / len(df) * 100
                    
                    st.success(f"✅ {after_final} rows added to `instagram_data` (skipped {before - after_final} duplicates).")
                    st.info(f"📊 Calculated Metrics: Avg Content Score: {avg_score:.3f}, Avg Engagement: {avg_engagement:.2f}%")
                    st.info(f"🎵 Artist Match Rate: {artist_match_rate:.1f}% of posts matched with uploaded artist list")
                    
                except SQLAlchemyError as e:
                    st.error(f"❌ Failed to upload {file.name}: {e}")
            else:
                st.info(f"⚠️ No new rows to upload from {file.name}.")

        except pd.errors.EmptyDataError:
            st.warning(f"⚠️ {file.name} is empty.")
        except SQLAlchemyError as e:
            st.error(f"❌ Database error for {file.name}: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error for {file.name}: {e}")

def process_artist_list(df, filename, engine):
    """Process and upload artist list to database"""
    try:
        # Standardize column names
        column_mapping = {
            'artist name': 'artist_name',
            'artist_name': 'artist_name',
            'genre': 'genre',
            'secondary genre': 'secondary_genre',
            'secondary_genre': 'secondary_genre'
        }
        
        # Rename columns to standard names
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure required columns exist
        if 'artist_name' not in df.columns:
            st.error("❌ Artist list must contain 'Artist Name' column")
            return
        
        # Fill missing optional columns
        if 'genre' not in df.columns:
            df['genre'] = 'Unknown'
        if 'secondary_genre' not in df.columns:
            df['secondary_genre'] = ''
        
        # Clean data
        df['artist_name'] = df['artist_name'].astype(str).str.strip()
        df['genre'] = df['genre'].astype(str).str.strip()
        df['secondary_genre'] = df['secondary_genre'].astype(str).str.strip()
        
        # Remove empty rows
        df = df[df['artist_name'].str.len() > 0]
        df = df[df['artist_name'] != 'nan']
        
        if df.empty:
            st.warning("⚠️ No valid artist data found in file")
            return
        
        # Remove duplicates
        before = len(df)
        df.drop_duplicates(subset=['artist_name'], inplace=True)
        after = len(df)
        
        # Remove duplicates already in database
        try:
            with engine.begin() as conn:
                # Check if artist_list table exists
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'artist_list')"
                ))
                table_exists = result.scalar()
                
                if table_exists:
                    existing_artists = pd.read_sql("SELECT artist_name FROM artist_list", conn)
                    df = df[~df['artist_name'].isin(existing_artists['artist_name'])]
        except SQLAlchemyError:
            # Table doesn't exist yet, will be created
            pass
        
        final_count = len(df)
        
        # Upload to artist_list table - ONLY include the columns we want
        if final_count > 0:
            # Select only the columns we need: artist_name, genre, secondary_genre
            df_to_upload = df[['artist_name', 'genre', 'secondary_genre']].copy()
            
            df_to_upload.to_sql(
                "artist_list",
                con=engine,
                if_exists="append",
                index=False,
                method="multi"
            )
            
            st.success(f"🎵 Added {final_count} new artists to artist list (skipped {before - final_count} duplicates)")
            st.info(f"📝 Total artists in database: {before - after + final_count}")
            
            # Show sample of uploaded artists
            with st.expander("View uploaded artists"):
                st.dataframe(df_to_upload.head(10))
        else:
            st.info("ℹ️ All artists in this file are already in the database")
            
    except Exception as e:
        st.error(f"❌ Error processing artist list: {str(e)}")

def process_demographic_files(uploaded_files):
    """Process updated Instagram demographic CSVs with city/country percentages."""
    engine = get_engine()

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for file in uploaded_files:
        try:
            fname = getattr(file, "name", getattr(file, "filename", "uploaded_file"))
            st.subheader(f"📊 Processing demographic data: {fname}")

            # Read raw CSV (no headers)
            # NOTE: file.getvalue().decode() is correct for Streamlit uploaded file objects
            text = file.getvalue().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(text), header=None).fillna("")

            # --- Detect sections ---
            sections = []
            i = 0
            while i < len(df):
                val = str(df.iloc[i, 0]).strip()
                if val and not re.match(r"^\d", val):  # not numeric, likely a section name
                    section_name = val
                    header_row = None

                    for j in range(i + 1, min(i + 5, len(df))):
                        # Use .any() to check if ANY value in the row is non-empty
                        if df.iloc[j].astype(bool).any():
                            header_row = j
                            break

                    if header_row is None:
                        i += 1
                        continue

                    next_section = None
                    for k in range(header_row + 1, len(df)):
                        next_val = str(df.iloc[k, 0]).strip()
                        if next_val and not re.match(r"^\d", next_val):
                            next_section = k
                            break

                    is_horizontal = any(s in section_name.lower() for s in ["citi", "countr"])
                    
                    if is_horizontal:
                        # For horizontal data, only grab the single data row (percentages)
                        section_df = df.iloc[header_row + 1:header_row + 2].copy()
                    else:
                        # For Age/Gender, grab all data rows until the next section
                        section_df = df.iloc[header_row + 1:next_section].copy() if next_section else df.iloc[header_row + 1:].copy()

                    # Set columns
                    section_df.columns = df.iloc[header_row].astype(str).tolist()
                    
                    # Clean up: drop fully empty rows/columns
                    section_df = section_df.replace({"": np.nan}).dropna(how="all").dropna(axis=1, how="all")

                    if not section_df.empty:
                        sections.append((section_name, section_df))
                    i = next_section if next_section else len(df)
                else:
                    i += 1

            if not sections:
                st.warning("⚠️ No valid sections found.")
                continue

            # --- Normalize + insert each section ---
            for section_name, section_df in sections:
                normalized_df, table_name = normalize_instagram_audience(section_name, section_df)
                if normalized_df is not None and not normalized_df.empty:
                    # NOTE: Pass DataFrame directly to insertion function to avoid CSV-related issues
                    create_and_insert_table(table_name, normalized_df)

        except Exception as e:
            st.error(f"❌ Error processing {fname}: {e}")


def normalize_instagram_audience(section_name: str, df: pd.DataFrame):
    """
    Normalize Instagram audience CSV sections for database insertion.
    FIXED: Logic for Top Cities/Countries to handle horizontal data spread.
    """
    print(f"Normalizing section: {section_name}")
    print(f"Raw DataFrame:\n{df.head(16)}")
    name = section_name.strip().lower()

    # --- Age & Gender section (Correct as-is for vertical data) ---
    if "age" in name:
        df = df.iloc[:, :3].copy()
        df.columns = ["age_range", "men", "women"]
        df["age_range"] = df["age_range"].astype(str).str.strip()
        df = df[df["age_range"] != ""]
        df["men"] = pd.to_numeric(df["men"].astype(str).str.replace(r'[^\d.]', '', regex=True), errors="coerce")
        df["women"] = pd.to_numeric(df["women"].astype(str).str.replace(r'[^\d.]', '', regex=True), errors="coerce")
        
        df = df.dropna(subset=["age_range", "men", "women"], how='all')
        
        return df, "demographics_followers_age_gender"

    # --- Top Cities (FIXED for horizontal data) ---
    elif "citi" in name:
        # After the fix in process_demographic_files, this DataFrame should be (1, N)
        if df.shape[0] == 1 and df.shape[1] > 1:
            df = df.T.reset_index()
            df.columns = ["city", "followers_percent"]
        else:
            # Fallback for unexpected or empty structure
            print(f"⚠️ City section has unexpected structure: {df.shape}. Skipping complex manual split.")
            return None, None
            
        # Clean
        df["city"] = df["city"].astype(str).str.strip()
        df["followers_percent"] = pd.to_numeric(df["followers_percent"], errors="coerce")
        df = df.dropna(subset=["city", "followers_percent"])
        return df, "demographics_followers_top_cities"

    # --- Top Countries (SIMPLIFIED and CORRECTED) ---
    elif "countr" in name:
        # After the fix in process_demographic_files, this DataFrame should be (1, N)
        if df.shape[0] == 1 and df.shape[1] > 1:
            df = df.T.reset_index()
            df.columns = ["country", "followers_percent"]
        else:
            # Fallback for unexpected or empty structure
            print(f"⚠️ Country section has unexpected structure: {df.shape}. Skipping complex manual split.")
            return None, None

        # Clean
        df["country"] = df["country"].astype(str).str.strip()
        df["followers_percent"] = pd.to_numeric(df["followers_percent"], errors="coerce")
        df = df.dropna(subset=["country", "followers_percent"])
        return df, "demographics_followers_top_countries"

    else:
        print(f"⚠️ Unknown section '{section_name}', skipping.")
        return None, None
    
def create_and_insert_table(table_name: str, df: pd.DataFrame):
    """
    Drop the existing table (if it exists) and insert the new DataFrame data.
    This implements the "replace all data" logic for new file uploads.
    """
    engine: Engine = get_engine()

    try:
        if df.empty:
            print(f"⚠️ Skipped empty table: {table_name}")
            st.warning(f"⚠️ Skipped empty table: {table_name}")
            return

        # Normalize names
        table_name = re.sub(r'\W+', '_', table_name.strip().lower())
        df.columns = [re.sub(r'\W+', '_', c.strip().lower()) for c in df.columns]

        # Clean string cells and replace NaN
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x).replace({np.nan: None})

        # Detect numeric columns
        numeric_cols = []
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            non_na_ratio = converted.notna().sum() / max(1, len(converted))
            if non_na_ratio >= 0.5:
                df[col] = converted
                numeric_cols.append(col)

        quoted_table = f'"{table_name}"'
        
        # --- START REPLACEMENT LOGIC ---
        with engine.begin() as conn:
            
            # 1. DROP THE OLD TABLE: This is the core change to implement replacement.
            drop_table_sql = f'DROP TABLE IF EXISTS {quoted_table} CASCADE;'
            conn.execute(text(drop_table_sql))
            print(f"🔄 Dropped existing table: {table_name}")

            # 2. CREATE THE NEW TABLE
            col_defs = []
            for col in df.columns:
                dtype = "DOUBLE PRECISION" if col in numeric_cols else "TEXT"
                # For age/country/city key columns, it's good practice to ensure they are NOT NULL
                is_key_col = col == df.columns[0]
                not_null = " NOT NULL" if is_key_col else ""
                col_defs.append(f'"{col}" {dtype}{not_null}')
            
            # Use the first column as the Primary Key for better database integrity
            key_col = df.columns[0]
            quoted_key = f'"{key_col}"'

            create_table_sql = f"""
            CREATE TABLE {quoted_table} (
                {', '.join(col_defs)},
                PRIMARY KEY ({quoted_key})
            );
            """
            conn.execute(text(create_table_sql))
            
            # 3. INSERT ALL DATA (using pandas to_sql for simplicity)
            df.to_sql(table_name, con=conn, if_exists="append", index=False, method="multi")
            
        # --- END REPLACEMENT LOGIC ---

        print(f"✅ Inserted {len(df)} rows into '{table_name}' (old data replaced).")
        st.success(f"✅ Inserted {len(df)} rows into '{table_name}' (old data replaced).")
        with st.expander(f"View sample of {table_name}"):
            st.dataframe(df.head(10))

    except Exception as e:
        print(f"❌ Error inserting into '{table_name}': {e}")
        st.error(f"❌ Error inserting into '{table_name}': {e}")

def process_followers_files(uploaded_files):
    """Process and upload followers data files to database"""
    engine = get_engine()
    
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    print(f"uploaded followers files: {uploaded_files}")
    for file in uploaded_files:
        try:
            # Get file name safely
            if hasattr(file, "name"):
                fname = file.name
            else:
                fname = getattr(file, "filename", "uploaded_file")
            print(f"fname: {fname}")
            st.subheader(f"📈 Processing followers data: {fname}")
            
            # Read the file
            if fname.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                # For CSV files, handle potential encoding issues
                try:
                    df = pd.read_csv(file)
                except UnicodeDecodeError:
                    # Try with different encoding if needed
                    df = pd.read_csv(file, encoding='utf-8')
            
            # Standardize column names
            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
            print(f"Columns in followers file: {df.columns.tolist()}")
            
            # Process followers data based on expected structure
            process_followers_data(df, fname, engine)
            
        except pd.errors.EmptyDataError:
            st.warning(f"⚠️ {fname} is empty.")
        except Exception as e:
            st.error(f"❌ Error processing {fname}: {str(e)}")
            print(f"Detailed error: {e}")

def process_followers_data(df, filename, engine):
    """Process and upload followers data to database"""
    try:
        # Standardize column names for followers data
        column_mapping = {
            'date': 'date',
            'primary': 'followers_count',
            'followers': 'followers_count',
            'value': 'followers_count',
            'count': 'followers_count'
        }
        
        # Rename columns to standard names
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Ensure required columns exist
        if 'date' not in df.columns:
            st.error("❌ Followers data must contain 'Date' column")
            return
            
        if 'followers_count' not in df.columns:
            st.error("❌ Followers data must contain a value column ('Primary', 'Followers', 'Value', or 'Count')")
            return
        
        # Clean and convert data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['followers_count'] = pd.to_numeric(df['followers_count'], errors='coerce')
        
        # Remove rows with invalid dates or counts
        df = df.dropna(subset=['date', 'followers_count'])
        print(f"Followers data after cleaning: {len(df)} rows")
        
        if df.empty:
            st.warning("⚠️ No valid followers data found in file")
            return
        
        # Add metadata
        df['source_file'] = filename
        df['processed_at'] = datetime.now()
        
        # Ensure we have the required columns for the database
        required_columns = ['date', 'followers_count', 'source_file', 'processed_at']
        for col in required_columns:
            if col not in df.columns:
                if col == 'followers_count':
                    df[col] = 0
                else:
                    df[col] = ''
        
        # Remove duplicates based on date
        before = len(df)
        df = df.drop_duplicates(subset=['date'])
        after = len(df)
        
        # Remove duplicates already in database
        try:
            with engine.begin() as conn:
                # Check if followers_metric_data table exists
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'followers_metric_data')"
                ))
                table_exists = result.scalar()
                
                if table_exists:
                    existing_dates = pd.read_sql("SELECT date FROM followers_metric_data", conn)
                    if not existing_dates.empty:
                        existing_dates['date'] = pd.to_datetime(existing_dates['date'])
                        df = df[~df['date'].isin(existing_dates['date'])]
        except Exception as e:
            print(f"Error checking existing data: {e}")
            # Table doesn't exist yet, will be created
        
        final_count = len(df)
        
        # Upload to followers_metric_data table
        if final_count > 0:
            # Ensure the table exists with proper schema
            with engine.begin() as conn:
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
                
                # Create index for better performance
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_followers_metric_data_date 
                    ON followers_metric_data(date)
                """))
            
            # Upload the data
            df_to_upload = df[['date', 'followers_count', 'source_file', 'processed_at']].copy()
            df_to_upload['date'] = df_to_upload['date'].dt.date  # Store as date only
            
            df_to_upload.to_sql(
                "followers_metric_data",
                con=engine,
                if_exists="append",
                index=False,
                method="multi"
            )
        else:
            st.info("ℹ️ All followers data in this file is already in the database")
            
    except Exception as e:
        st.error(f"❌ Error processing followers data: {str(e)}")
        print(f"Detailed error in process_followers_data: {e}")
    

def get_instagram_data(engine, limit: int = None):
    """Retrieve data from the unified instagram_data table"""
    try:
        query = "SELECT * FROM instagram_data"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error retrieving instagram_data: {e}")
        return pd.DataFrame()

def get_artist_list_data(engine):
    """Retrieve artist list from database"""
    try:
        return pd.read_sql("SELECT * FROM artist_list", engine)
    except Exception as e:
        print(f"Error retrieving artist_list: {e}")
        return pd.DataFrame()
    