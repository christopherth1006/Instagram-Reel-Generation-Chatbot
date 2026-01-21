"""
AI Music Curator Assistant - Streamlit Web Application
Optimized version with performance improvements and warning fixes
"""
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import text

from utils.analytics_engine import AnalyticsEngine
from generators.content_generator import ContentGenerator
from utils.export_manager import ExportManager
from utils.artist_list_manager import ArtistListManager
from chat.chat_pipeline import ChatPipeline
from chat.chat_interface import ChatInterface
from config import STREAMLIT_CONFIG, OPENAI_API_KEY, BRAND_VOICE
from utils.process_upload import process_uploaded_files
from utils.process_upload import process_demographic_files
from utils.process_upload import process_followers_files
from utils.process_upload import get_engine
from utils.postgres_manager import PostgresManager

# Set page config first
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS with optimized styles
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .content-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .export-section {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        .metric-card {
            margin: 0.25rem 0;
        }
    }
    
    /* Table alignment styles */
    .stDataFrame table {
        text-align: center !important;
    }
    
    .stDataFrame table th {
        text-align: center !important;
    }
    
    .stDataFrame table td {
        text-align: center !important;
    }
    
    /* Left align description columns */
    .stDataFrame table td:nth-child(1) {
        text-align: left !important;
    }
    
    /* Center align subheaders */
    .stSubheader {
        text-align: center !important;
    }
    
    /* Chat-style input styling */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #667eea30 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Primary button styling */
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton button[kind="primary"]:active {
        transform: translateY(0) !important;
    }
</style>
""", unsafe_allow_html=True)

class MusicCuratorApp:
    def __init__(self):
        self._initialize_session_state()
        
        self.analytics_engine = AnalyticsEngine()
        self.export_manager = ExportManager()
        self.db_engine = get_engine()
        self.postgres_manager = PostgresManager(db_engine = self.db_engine)  
        self.artist_list_manager = ArtistListManager(db_engine=self.db_engine)
        self.content_generator = None

        if self.check_api_key():
            try:
                self.content_generator = ContentGenerator(
                    OPENAI_API_KEY, 
                    self.artist_list_manager,
                    self.db_engine
                )
            except Exception as e:
                self.content_generator = None

        self.chat_pipeline = ChatPipeline(
            db_engine=self.db_engine,
            content_generator=self.content_generator,
            artist_list_manager=self.artist_list_manager
        )
        self.chat_interface = ChatInterface(self.chat_pipeline)

    def _initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'analytics_data': None,
            'insights': None,
            'generated_content': {},
            'generating_section': None,
            '_pending_reels': False,
            '_pending_quizzes': False,
            '_pending_polls': False,
            'error_message': None,
            'error_timestamp': None,
            'loaded_files': {},
            'file_dataframes': {},
            'postgres_has_data': False,
            'postgres_initialized': False,
            'last_postgres_update': None,
            'previous_uploaded_files': set(),
            'app_initialized': True
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _check_postgres_status(self):
        """Check PostgreSQL status and load data if available"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'instagram_data'
                    )
                """))
                table_exists = result.scalar()
                
                if table_exists:
                    result = conn.execute(text("SELECT COUNT(*) FROM instagram_data"))
                    post_count = result.scalar()
                    st.session_state.postgres_has_data = post_count > 0
                    st.session_state.postgres_posts_count = post_count
                    
                    if post_count > 0 and not st.session_state.insights:
                        self._refresh_postgres_insights()
                else:
                    st.session_state.postgres_has_data = False
                    st.session_state.postgres_posts_count = 0
                    
        except Exception as e:
            st.session_state.postgres_has_data = False
            st.session_state.postgres_posts_count = 0

    def _refresh_postgres_insights(self):
        """Refresh insights from PostgreSQL database"""
        try:
            insights = self._get_comprehensive_insights_from_postgres()
            st.session_state.insights = insights
            st.session_state.postgres_has_data = insights.get('total_posts', 0) > 0
            st.session_state.last_postgres_update = datetime.now()
            
            # Update chat pipeline if it exists
            if self.chat_pipeline:
                self.chat_pipeline.update_analytics_data(None, insights)
                
        except Exception as e:
            print(f"Error refreshing PostgreSQL insights: {e}")

    def _get_comprehensive_insights_from_postgres(self) -> Dict:
        """Extract comprehensive insights from PostgreSQL database"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_posts,
                        AVG(ai_post_score) as avg_content_score,
                        MAX(ai_post_score) as top_content_score,
                        AVG(engagement_rate_impr) as avg_engagement_rate,
                        AVG(follow_conversion) as avg_follow_conversion
                    FROM instagram_data
                """))
                stats = result.fetchone()
                print(f"stats: {stats}")
                
                if not stats or stats[0] == 0:
                    return {'total_posts': 0}
                
                insights = {
                    'total_posts': stats[0],
                    'avg_content_score': float(stats[1] or 0),
                    'top_content_score': float(stats[2] or 0),
                    'avg_engagement_rate': float(stats[3] or 0) / 100,
                    'avg_follows_per_impression': float(stats[4] or 0) / 100
                }
                
                result = conn.execute(text("""
                    SELECT 
                        hook_type,
                        AVG(ai_post_score) as avg_score,
                        COUNT(*) as count
                    FROM instagram_data 
                    WHERE hook_type IS NOT NULL AND hook_type != '' AND ai_post_score > 0
                    GROUP BY hook_type
                    ORDER BY avg_score DESC
                """))
                
                hook_performance = {}
                for row in result:
                    hook_performance[row[0]] = float(row[1] or 0)
                
                insights['hook_performance'] = {
                    'avg_score': hook_performance
                }
                
                result = conn.execute(text("""
                    SELECT 
                        time_bucket,
                        AVG(ai_post_score) as avg_score,
                        COUNT(*) as count
                    FROM instagram_data 
                    WHERE time_bucket IS NOT NULL AND time_bucket != '' AND ai_post_score > 0
                    GROUP BY time_bucket
                    ORDER BY avg_score DESC
                """))
                
                time_performance = {}
                for row in result:
                    time_performance[row[0]] = float(row[1] or 0)
                
                insights['time_performance'] = {
                    'avg_score': time_performance
                }
                
                result = conn.execute(text("""
                    SELECT 
                        source_file,
                        COUNT(*) as post_count,
                        AVG(ai_post_score) as avg_score,
                        MAX(ai_post_score) as top_score,
                        AVG(engagement_rate_impr) as avg_engagement,
                        AVG(follow_conversion) as avg_follow_conversion
                    FROM instagram_data 
                    WHERE source_file IS NOT NULL AND source_file != ''
                    GROUP BY source_file
                    ORDER BY post_count DESC
                """))
                
                source_files = {}
                for row in result:
                    source_files[row[0]] = {
                        'post_count': row[1],
                        'avg_score': float(row[2] or 0),
                        'top_score': float(row[3] or 0),
                        'avg_engagement': float(row[4] or 0) / 100,
                        'avg_follows_per_impression': float(row[5] or 0) / 100
                    }
                
                insights['source_files'] = source_files
                
                return insights
                
        except Exception as e:
            print(f"❌ Error getting comprehensive insights from PostgreSQL: {e}")
            return {'total_posts': 0}
        
    def _get_top_posts_from_postgres(self, n: int = 10) -> List[Dict]:
        """Get top N posts from PostgreSQL."""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT
                        post_id,
                        description,
                        permalink,
                        ai_post_score AS content_score,
                        hook_type,
                        engagement_rate_impr AS engagement_rate_pct,
                        follow_conversion AS follow_rate_fraction
                    FROM instagram_data
                    WHERE ai_post_score > 0
                    AND follow_conversion IS NOT NULL
                    ORDER BY ai_post_score DESC
                    LIMIT :n
                """), {"n": n})

                top_posts = []
                for row in result:
                    post_id, desc, permalink, score, hook, er_pct, fr_frac = row

                    engagement_rate = float(er_pct or 0) / 100.0

                    follows_per_impression = float(fr_frac or 0)

                    top_posts.append({
                        "post_id": post_id,
                        "description": desc,
                        "permalink": permalink,
                        "content_score": float(score or 0),
                        "hook_type": hook,
                        "engagement_rate": engagement_rate,
                        "follows_per_impression": follows_per_impression
                    })

                return top_posts

        except Exception as e:
            print(f"❌ Error getting top posts from PostgreSQL: {e}")
            return []
        
    def _get_score_distribution_from_postgres(self) -> List[float]:
        """Get content score distribution from PostgreSQL"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT ai_post_score 
                    FROM instagram_data 
                    WHERE ai_post_score > 0
                """))
                
                scores = [float(row[0]) for row in result if row[0] is not None]
                return scores
                
        except Exception as e:
            print(f"❌ Error getting score distribution from PostgreSQL: {e}")
            return []

    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        if st.session_state.error_message:
            # Use a persistent container for the error
            if 'error_container' not in st.session_state:
                st.session_state.error_container = st.empty()
            
            st.session_state.error_container.error(f"❌ {st.session_state.error_message}")
        
        self.clear_error_if_expired()

        self._check_postgres_status()
        
        if not self.check_api_key():
            self.render_api_key_setup()
            return
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analytics & Insights", "🎬 Reel Generator", "❓ Quiz Generator", "🗳️ Poll Generator", "💬 Chat with Your Analytics"])
        
        with tab1:
            self.render_analytics_tab()
        
        with tab2:
            self.render_reels_tab()
        
        with tab3:
            self.render_quiz_tab()
        
        with tab4:
            self.render_polls_tab()

        with tab5:
            self.render_chat_tab()
    
    def render_header(self):
        """Render application header"""
        st.markdown(f"""
        <div class="main-header">
            <h1>🎵 {BRAND_VOICE.get('persona_name', 'AI Music Curator')} Assistant</h1>
            <p>Analytics • Content Generation • Growth Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:

            if "analytics_uploader_key" not in st.session_state:
                st.session_state.analytics_uploader_key = 0

            if "demographic_uploader_key" not in st.session_state:
                st.session_state.demographic_uploader_key = 0

            if "followers_uploader_key" not in st.session_state:
                st.session_state.followers_uploader_key = 0

            st.header("📤 Upload Analytics Data")

            uploaded_files = st.file_uploader(
                "Upload IG Analytics CSV(s)",
                type=['csv'],
                help="Upload your Instagram analytics CSV exports (post file or story file).",
                accept_multiple_files=True,
                key=f"analytics_uploader_{st.session_state.analytics_uploader_key}"
            )

            if uploaded_files:
                process_uploaded_files(uploaded_files)

                st.session_state.analytics_uploader_key += 1
                st.rerun()

            st.header("📤 Upload Demographic Data")

            demographic_uploaded_files = st.file_uploader(
                "Upload Demographic Data CSV(s)",
                type=['csv', 'xls', 'xlsx'],
                help="Upload your Demographic CSV Data exports.",
                accept_multiple_files=True,
                key=f"demographic_uploader_{st.session_state.demographic_uploader_key}"
            )

            if demographic_uploaded_files:
                process_demographic_files(demographic_uploaded_files)

                st.session_state.demographic_uploader_key += 1
                st.rerun()

            st.header("📤 Upload Followers Data")

            followers_uploaded_files = st.file_uploader(
                "Upload Followers Data CSV(s)",
                type=['csv'],
                help="Upload your Followers CSV Data exports.",
                accept_multiple_files=True,
                key=f"followers_uploader_{st.session_state.followers_uploader_key}"
            )

            if followers_uploaded_files:
                process_followers_files(followers_uploaded_files)

                st.session_state.followers_uploader_key += 1
                st.rerun()

            st.header("🎤 Brand Voice")
            st.info(f"""
            **Persona**: {BRAND_VOICE.get('persona_name', 'Music Curator')}
            **Vibe**: {BRAND_VOICE.get('vibe', 'Music curation')}
            **Tone**: {', '.join(BRAND_VOICE.get('tone', []))}
            """)
            
            self.render_artist_list_section()
            
            st.header("📤 Quick Export")
            if st.button("Export All Data", use_container_width=True, type="primary"):
                with st.spinner("Preparing all data for export..."):
                    self.export_all_data()

    def check_api_key(self) -> bool:
        """Check if OpenAI API key is available"""
        return OPENAI_API_KEY is not None and len(OPENAI_API_KEY) > 0
    
    def show_error_with_delay(self, error_message: str, delay_seconds: int = 5):
        """Show error message with automatic dismissal after delay"""
        st.session_state.error_message = error_message
        st.session_state.error_timestamp = time.time()
        st.session_state.error_duration = delay_seconds
        error_container = st.empty()
        st.error(f"❌ {error_message}")
        
        st.session_state.error_container = error_container
    
    def clear_error_if_expired(self):
        """Clear error message if it has expired"""
        if (st.session_state.error_message and 
            st.session_state.error_timestamp and 
            time.time() - st.session_state.error_timestamp >= st.session_state.get('error_duration', 10)):
            
            # Clear the error container if it exists
            if 'error_container' in st.session_state and st.session_state.error_container:
                st.session_state.error_container.empty()
            
            st.session_state.error_message = None
            st.session_state.error_timestamp = None
            st.session_state.error_duration = 10  # Reset to default
    
    def render_api_key_setup(self):
        """Render API key setup interface"""
        st.error("⚠️ OpenAI API Key Required")
        st.markdown("""
        To use the content generation features, please set your OpenAI API key:
        
        1. Create a `.env` file in the project root
        2. Add: `OPENAI_API_KEY=your_key_here`
        3. Restart the application
        
        Or set it as an environment variable: `export OPENAI_API_KEY=your_key_here`
        """)
    
    def render_analytics_tab(self):
        """Render analytics and insights tab using cached data"""
        st.header("📊 Analytics & Insights")
        has_data = (st.session_state.postgres_has_data and 
                   st.session_state.insights and 
                   st.session_state.insights.get('total_posts', 0) > 0)
        
        if not has_data:
            st.info("👆 Upload your Instagram analytics CSV to get started")
            return
        
        if not self.check_api_key():
            st.warning("⚠️ OpenAI API key required for content generation")
            return
        
        insights = st.session_state.insights or {}

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_posts = insights.get('total_posts', 0)
            st.metric("Total Posts", f"{total_posts:,}")

        with col2:
            avg_score = insights.get('avg_content_score', 0)
            st.metric("Avg ContentScore", f"{avg_score:.2f}")

        with col3:
            top_score = insights.get('top_content_score', 0)
            st.metric("Top Score", f"{top_score:.2f}")

        with col4:
            avg_engagement_rate = insights.get('avg_engagement_rate', 0)
            st.metric("Engagement Rate", f"{avg_engagement_rate * 100:.2f}%")
        
        if st.button("🔄 Refresh Analytics Data", type="secondary"):
            with st.spinner("Refreshing from Postgres..."):
                self._refresh_postgres_insights()
                st.success("✅ Analytics data refreshed!")
                st.rerun()
        
        if 'source_files' in insights and len(insights['source_files']) > 1:
            st.subheader("📁 Dataset Breakdown")
            file_breakdown_data = []
            
            for filename, file_info in insights['source_files'].items():
                file_breakdown_data.append({
                    'Source File': filename,
                    'Posts': file_info.get('post_count', 0),
                    'Avg Score': f"{file_info.get('avg_score', 0):.2f}",
                    'Top Score': f"{file_info.get('top_score', 0):.2f}",
                    'Avg Engagement': f"{file_info.get('avg_engagement', 0) * 100:.2f}%"
                })
            
            file_breakdown = pd.DataFrame(file_breakdown_data)
            file_breakdown = file_breakdown.sort_values('Posts', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(file_breakdown, use_container_width=True)
            with col2:
                st.info(f"""
                **Files Processed**: {len(insights['source_files'])}
                
                **Total Unique Posts**: {total_posts:,}
                """)
        
        st.subheader("🏆 Top Performing Posts")

        top_posts = self._get_top_posts_from_postgres(50)

        if top_posts:
            df = pd.DataFrame(top_posts)

            df['description'] = df['description'].str.replace('\n', ' ', regex=False)

            df.insert(0, 'No', range(1, len(df) + 1))

            st.dataframe(
                df,
                column_config={
                    'No': 'No',
                    'description': 'Description',
                    "permalink": st.column_config.LinkColumn(
                        "Link",
                        display_text="🔗",
                        width='small',
                    ),
                    "content_score": st.column_config.NumberColumn("Content Score"),
                    'hook_type': 'Hook Type',
                    "engagement_rate": st.column_config.NumberColumn("Engagement Rate", format="%.3f"),
                    "follows_per_impression": st.column_config.NumberColumn("Follows/Impression", format="%.4f"),
                },

                hide_index=True,
                column_order=[
                    'No',
                    "description",
                    "permalink",
                    "content_score",
                    "hook_type",
                    "engagement_rate",
                    "follows_per_impression",
                ],

                height=500
            )
        else:
            st.info("No top posts data available")
            
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 ContentScore Distribution")

            scores = self._get_score_distribution_from_postgres()

            if scores:
                try:
                    scores = [float(score) for score in scores]
                except (ValueError, TypeError) as e:
                    st.error(f"Error converting scores to numeric: {e}")
                    st.write("Sample of problematic scores:", [score for score in scores[:5] if not isinstance(score, (int, float))])
                    return
                
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                if score_range <= 10:
                    bin_size = 1
                elif score_range <= 20:
                    bin_size = 2
                elif score_range <= 50:
                    bin_size = 5
                else:
                    bin_size = 10
                
                start_bin = max(0, np.floor(min_score / bin_size) * bin_size)
                end_bin = (np.ceil(max_score / bin_size) + 1) * bin_size
                
                bins = np.arange(start_bin, end_bin + 0.1, bin_size)
                
                hist, bin_edges = np.histogram(scores, bins=bins)
                
                bin_labels = []
                for i in range(len(bin_edges)-1):
                    if bin_size == 1:
                        bin_labels.append(f"{bin_edges[i]:.1f}")
                    else:
                        bin_labels.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}")
                
                score_df = pd.DataFrame({
                    'Score Range': bin_labels,
                    'Number of Posts': hist
                })
                
                score_df = score_df[score_df['Number of Posts'] > 0]
                
                if len(score_df) == 0:
                    st.info("No data to display in histogram")
                    return

                fig = px.bar(
                    score_df,
                    x='Score Range',
                    y='Number of Posts',
                    color='Number of Posts',
                    color_continuous_scale='blues',
                    title=f"Distribution of {len(scores)} Posts"
                )

                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Content Score Range",
                    yaxis_title="Number of Posts",
                    margin=dict(l=40, r=40, t=60, b=40),
                    xaxis={'type': 'category'}
                )

                fig.update_coloraxes(showscale=False)

                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.info("📊 No score distribution data available")
                

        with col2:
            # Fix the hook_performance display
            st.subheader("🎯 Hook Type Performance")
            if 'hook_performance' in insights and insights['hook_performance']:
                # Check the structure and extract the average scores
                hook_data = insights['hook_performance'].get('avg_score', {})
                
                # Filter out zero values and sort by score
                filtered_hook_data = {k: v for k, v in hook_data.items() if v > 0}
                
                if filtered_hook_data:
                    hook_df = pd.DataFrame({
                        'Hook Type': list(filtered_hook_data.keys()),
                        'Average Score': list(filtered_hook_data.values())
                    }).sort_values('Average Score', ascending=False)
                    
                    fig = px.bar(
                        hook_df,
                        x='Hook Type',
                        y='Average Score',
                        color='Average Score',
                        color_continuous_scale='greens'
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        xaxis_title="Hook Type",
                        yaxis_title="Average Content Score",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                else:
                    st.info("🎯 No hook performance data available")
            else:
                st.info("🎯 No hook performance data available")

        st.subheader("⏰ Performance by Time of Day")
        if 'time_performance' in insights and insights['time_performance']:
            time_data = insights['time_performance'].get('avg_score', {})
            
            # Define exact order for time buckets
            time_order = ["Morning (6a-12p)", "Afternoon (12p-6p)", "Evening (6p-12a)", "Night (12a-6a)"]
            
            # Create DataFrame in exact order
            time_df = pd.DataFrame({
                'Time of Day': time_order,
                'Average Score': [time_data.get(time_key, 0.0) for time_key in time_order]
            })
            
            # Only show if we have at least one non-zero value
            if any(time_df['Average Score'] > 0):
                fig = px.bar(
                    time_df,
                    x='Time of Day',
                    y='Average Score',
                    color='Average Score',
                    color_continuous_scale='purples'
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Time of Day",
                    yaxis_title="Average Content Score",
                    xaxis={'categoryorder': 'array', 'categoryarray': time_order},
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.info("⏰ No time performance data available")
        else:
            st.info("⏰ No time performance data available")

        st.subheader("📈 Followers Analytics")
        
        try:
            # Get followers data
            followers_df = self.postgres_manager.get_followers_data()
            
            if not followers_df.empty:
                # Calculate basic metrics
                current_followers = followers_df['followers_count'].iloc[-1] if len(followers_df) > 0 else 0
                previous_followers = followers_df['followers_count'].iloc[-2] if len(followers_df) > 1 else current_followers
                daily_growth = current_followers - previous_followers
                
                total_growth = followers_df['followers_count'].iloc[-1] - followers_df['followers_count'].iloc[0]
                growth_percentage = (total_growth / followers_df['followers_count'].iloc[0] * 100) if followers_df['followers_count'].iloc[0] > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Followers", f"{current_followers:,}")
                
                with col2:
                    st.metric("Daily Growth", f"{daily_growth:+,}", delta=f"{daily_growth:+,}")
                
                with col3:
                    st.metric("Total Growth", f"{total_growth:+,}")
                
                with col4:
                    st.metric("Growth %", f"{growth_percentage:.1f}%")
                
                # Followers trend chart
                fig = px.line(
                    followers_df, 
                    x='date', 
                    y='followers_count',
                    title='Followers Growth Over Time',
                    labels={'followers_count': 'Followers', 'date': 'Date'}
                )
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Followers Count",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                
                # # Growth metrics for different periods
                # st.subheader("📊 Growth Metrics")
                
                # col1, col2, col3 = st.columns(3)
                
                # with col1:
                #     weekly_metrics = self.postgres_manager.get_followers_growth_metrics(7)
                #     if weekly_metrics:
                #         st.metric("7-Day Growth", 
                #                 f"{weekly_metrics.get('total_growth', 0):+}",
                #                 f"{weekly_metrics.get('growth_percentage', 0):.1f}%")
                
                # with col2:
                #     monthly_metrics = self.postgres_manager.get_followers_growth_metrics(30)
                #     if monthly_metrics:
                #         st.metric("30-Day Growth", 
                #                 f"{monthly_metrics.get('total_growth', 0):+}",
                #                 f"{monthly_metrics.get('growth_percentage', 0):.1f}%")
                
                # with col3:
                #     st.metric("Overall Growth", 
                #             f"{total_growth:+}",
                #             f"{growth_percentage:.1f}%")
                
                # Export followers data
                with st.expander("📤 Export Followers Data"):
                    csv_data = self.export_manager.get_csv_download_link(followers_df)
                    st.download_button(
                        label="📈 Export Followers Data",
                        data=csv_data,
                        file_name="followers_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            else:
                st.info("👆 Upload followers data CSV to see analytics")
                
        except Exception as e:
            st.error(f"Error loading followers data: {str(e)}")
        
        # Export Section
        with st.expander("📤 Export Analytics"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Export Top Posts using SQL query
                try:
                    with self.db_engine.connect() as conn:
                        top_posts_query = text("""
                            SELECT 
                                post_id,
                                description,
                                publish_time,
                                permalink,
                                post_type,
                                engagements,
                                engagement_rate_impr,
                                follow_conversion,
                                dow,
                                hour_local,
                                time_bucket,
                                content_type_tags,
                                hook_type,
                                main_artists,
                                subgenre
                            FROM instagram_data 
                            WHERE ai_post_score > 0
                            ORDER BY ai_post_score DESC
                            LIMIT 50
                        """)
                        top_posts_df = pd.read_sql(top_posts_query, conn)
                        
                        if not top_posts_df.empty:
                            csv_data = self.export_manager.get_csv_download_link(top_posts_df)
                            st.download_button(
                                label="🏆 Export TOP POSTS",
                                data=csv_data,
                                file_name="analytics_top_posts.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.info("No top posts data available for export")
                except Exception as e:
                    st.error(f"Error exporting top posts: {str(e)}")
            
            with col2:
                # Export ALL Analytics using SQL query
                try:
                    with self.db_engine.connect() as conn:
                        all_data_query = text("""
                            SELECT 
                                post_id,
                                description,
                                publish_time,
                                permalink,
                                post_type,
                                engagements,
                                engagement_rate_impr,
                                follow_conversion,
                                dow,
                                hour_local,
                                time_bucket,
                                content_type_tags,
                                hook_type,
                                main_artists,
                                subgenre
                            FROM instagram_data 
                            ORDER BY ai_post_score DESC NULLS LAST, publish_time DESC
                        """)
                        all_data_df = pd.read_sql(all_data_query, conn)
                        
                        if not all_data_df.empty:
                            csv_data = self.export_manager.get_csv_download_link(all_data_df)
                            st.download_button(
                                label="📦 Export ALL Analytics",
                                data=csv_data,
                                file_name="analytics_all_data.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.info("No analytics data available for export")
                except Exception as e:
                    st.error(f"Error exporting all analytics: {str(e)}")
                
    def _check_postgres_has_data(self) -> bool:
        """Safely check if PostgreSQL has data"""
        self._check_postgres_status()
        return st.session_state.postgres_has_data

    def render_reels_tab(self):
        """Render Reel generator tab using cached data"""
        st.header("🎬 Reel Script Generator")
        
        has_data = (st.session_state.postgres_has_data and 
                   st.session_state.insights and 
                   st.session_state.insights.get('total_posts', 0) > 0)
        
        if not has_data:
            st.info("👆 Upload your Instagram analytics CSV to get started")
            return
        
        if not self.check_api_key():
            st.warning("⚠️ OpenAI API key required for content generation")
            return

        # Current tab name for logic
        current_tab = "reels"
        disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab

        # Show banner if controls are disabled because another generation is running
        if disabled:
            st.warning(f"⚠️ Controls disabled while `{st.session_state.generating_section}` generation is in progress.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # disable while another section generating
            batch_size = st.slider("Batch Size", 10, 20, 15, disabled=disabled)
        
        with col2:
            focus_themes = st.multiselect(
                "Focus Themes",
                ["Trending Artists", "Classic Hits", "New Releases", "Underground", "Throwback"],
                default=["Trending Artists"],
                disabled=disabled
            )
        
        # Start generation with a pending pattern so the UI updates to disabled state before heavy work
        generate_clicked = st.button("🎬 Generate Reel Scripts", type="primary", disabled=disabled)
        if generate_clicked:
            st.session_state.generating_section = "reels"
            st.session_state._pending_reels = True
            st.rerun()

        if st.session_state._pending_reels and st.session_state.generating_section == "reels":
            with st.spinner("Generating creative Reel scripts..."):
                try:
                    insights = st.session_state.insights or {}
                    
                    reels_df = self.content_generator.generate_reel_scripts(
                        insights,
                        batch_size=batch_size,
                        focus_themes=focus_themes,
                        analytics_data=None
                    )
                    st.session_state.generated_content['reels'] = reels_df
                    st.success(f"✅ Generated {len(reels_df)} Reel scripts!")
                except Exception as e:
                    self.show_error_with_delay(f"Reel generation failed: {str(e)}", delay_seconds=10)
                finally:
                    st.session_state._pending_reels = False
                    st.session_state.generating_section = None
                    st.rerun()
        
        # Display generated reels (filters and exports should be disabled if another generation running)
        if 'reels' in st.session_state.generated_content and st.session_state.generated_content['reels'] is not None:
            reels_df = st.session_state.generated_content['reels']
            
            st.subheader(f"📋 Generated Scripts ({len(reels_df)} total)")
            
            col1, col2 = st.columns(2)
            with col1:
                score_disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
                score_threshold = st.slider("Min Predicted Score", 0, 100, 70, disabled=score_disabled)
            with col2:
                artist_disabled = score_disabled
                if 'artist' in reels_df.columns:
                    # Filter artists to only show those from uploaded list
                    all_artists = reels_df['artist'].unique()
                    uploaded_artists = []
                    if self.artist_list_manager:
                        active_list = self.artist_list_manager.get_active_artist_list()
                        if active_list and 'artists' in active_list:
                            uploaded_artist_names = {artist['name'] for artist in active_list['artists']}
                            uploaded_artists = [artist for artist in all_artists if artist in uploaded_artist_names]
                    
                    # Use uploaded artists if available, otherwise fall back to all artists
                    available_artists = uploaded_artists if uploaded_artists else all_artists
                    
                    selected_artists = st.multiselect(
                        "Filter by Artist",
                        available_artists,
                        default=available_artists,
                        disabled=artist_disabled
                    )
                else:
                    selected_artists = []
            
            # Guard against empty DataFrame or missing column
            if 'predicted_score' not in reels_df.columns:
                st.warning("No reel items matched your artist filter. Try lowering filters or updating the artist list.")
                filtered_df = reels_df.copy()
            else:
                filtered_df = reels_df[reels_df['predicted_score'] >= score_threshold]
            if selected_artists and 'artist' in reels_df.columns:
                filtered_df = filtered_df[filtered_df['artist'].isin(selected_artists)]
            
            for idx, row in filtered_df.iterrows():
                with st.expander(f"🎵 {row.get('artist', 'Unknown')} - Score: {row.get('predicted_score', 0):.1f}"):
                    st.write("**Hook Text:**", row.get('hook_text', ''))
                    st.write("**Audio Suggestion:**", row.get('audio_suggestion', ''))
                    
                    if isinstance(row.get('captions'), list):
                        st.write("**Caption Options:**")
                        for i, caption in enumerate(row['captions'], 1):
                            st.write(f"{i}. {caption}")
                    
                    if isinstance(row.get('ctas'), list):
                        st.write("**CTA Options:**")
                        for i, cta in enumerate(row['ctas'], 1):
                            st.write(f"{i}. {cta}")
            
            with st.expander("📤 Export Reels"):
                # export button should also reflect disabled state
                formatted_df = self.export_manager.format_for_export(filtered_df, "reels")
                csv_data = self.export_manager.get_csv_download_link(formatted_df, "reel_scripts.csv")
                st.download_button(
                    "Download Reel Scripts CSV",
                    csv_data,
                    "reel_scripts.csv",
                    "text/csv",
                    disabled= st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
                )
    
    def render_quiz_tab(self):
        """Render Quiz generator tab using cached data"""
        st.header("❓ Story Quiz Generator")
        
        has_data = (st.session_state.postgres_has_data and 
                st.session_state.insights and 
                st.session_state.insights.get('total_posts', 0) > 0)
        
        if not has_data:
            st.info("👆 Upload your Instagram analytics CSV to get started")
            return
        
        if not self.check_api_key():
            st.warning("⚠️ OpenAI API key required for content generation")
            return

        current_tab = "quizzes"
        disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
        
        if disabled:
            st.warning(f"⚠️ Controls disabled while `{st.session_state.generating_section}` generation is in progress.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.slider("Quiz Batch Size", 50, 75, 60, disabled=disabled)
        
        with col2:
            quiz_types = st.multiselect(
                "Quiz Types",
                ["Who Said It", "Fill in the Blank", "Guess the Year", "Sample Match"],
                default=["Who Said It"],
                disabled=disabled
            )
        
        with col3:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard"],
                index=1,  # Default to Medium
                disabled=disabled,
                help="Easy: Mainstream artists & popular songs\nMedium: Mixed difficulty\nHard: Underground artists & deep cuts"
            )
        
        # Generate button - CLEAR previous quizzes when generating new ones
        generate_clicked = st.button("❓ Generate Quiz Questions", type="primary", disabled=disabled)
        if generate_clicked:
            st.session_state.generating_section = current_tab
            st.session_state._pending_quizzes = True
            st.rerun()
        
        if st.session_state._pending_quizzes and st.session_state.generating_section == current_tab:
            with st.spinner("Creating engaging quiz questions..."):
                try:
                    insights = st.session_state.insights or {}
                    
                    quiz_df = self.content_generator.generate_story_quizzes(
                        insights,
                        batch_size=batch_size,
                        quiz_types=quiz_types,
                        difficulty=difficulty.lower(),
                        chat_history=[]
                    )
                    st.session_state.generated_content['quizzes'] = quiz_df
                    # Clear previous selections when generating new quizzes
                    if 'selected_quizzes' in st.session_state:
                        st.session_state.selected_quizzes.clear()
                    st.success(f"✅ Generated {len(quiz_df)} {difficulty.lower()} quiz questions!")
                except Exception as e:
                    self.show_error_with_delay(f"Quiz generation failed: {str(e)}", delay_seconds=10)
                finally:
                    st.session_state._pending_quizzes = False
                    st.session_state.generating_section = None
                    st.rerun()
        
        if 'quizzes' in st.session_state.generated_content and st.session_state.generated_content['quizzes'] is not None:
            quiz_df = st.session_state.generated_content['quizzes']
                
            st.subheader(f"🧩 Generated {difficulty} Quizzes ({len(quiz_df)} total)")
            
            col1, col2 = st.columns(2)
            with col1:
                score_disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
                score_threshold = st.slider("Min Predicted Score", 0, 100, 0, key="quiz_score", disabled=score_disabled)
            with col2:
                type_disabled = score_disabled
                if 'type' in quiz_df.columns:
                    selected_types = st.multiselect(
                        "Filter by Type",
                        quiz_df['type'].unique(),
                        default=list(quiz_df['type'].unique()),
                        disabled=type_disabled
                    )
                else:
                    selected_types = []
            
            if 'predicted_score' not in quiz_df.columns:
                st.warning("No quiz items matched your filters. Showing all available.")
                display_df = quiz_df.copy()
            else:
                display_df = quiz_df[quiz_df['predicted_score'] >= score_threshold]
            
            if selected_types and 'type' in quiz_df.columns:
                display_df = display_df[display_df['type'].isin(selected_types)]
            
            # Initialize selected quizzes in session state
            if 'selected_quizzes' not in st.session_state:
                st.session_state.selected_quizzes = set()
            
            # Display filtered quizzes with selection checkboxes
            selected_indices = []
            for idx, row in display_df.iterrows():
                difficulty_badge = f" ({row.get('difficulty', 'medium').upper()})" if 'difficulty' in row else ""
                
                # Create a unique key for each quiz
                quiz_key = f"quiz_{idx}"
                
                # Checkbox for selection - use a unique key for the checkbox widget
                is_selected = st.checkbox(
                    f"Select this quiz - Score: {row.get('predicted_score', 0):.1f}",
                    value=quiz_key in st.session_state.selected_quizzes,
                    key=f"checkbox_{quiz_key}"  # Different key for the checkbox widget
                )
                
                if is_selected:
                    st.session_state.selected_quizzes.add(quiz_key)
                    selected_indices.append(idx)
                elif quiz_key in st.session_state.selected_quizzes:
                    st.session_state.selected_quizzes.remove(quiz_key)
                
                with st.expander(f"❓ {row.get('type', 'Quiz')}{difficulty_badge} - Score: {row.get('predicted_score', 0):.1f}"):
                    st.write("**Question:**", row.get('question', ''))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("A)", row.get('option_a', ''))
                        st.write("C)", row.get('option_c', ''))
                    with col2:
                        st.write("B)", row.get('option_b', ''))
                        st.write("D)", row.get('option_d', ''))
                    
                    st.success(f"**Correct Answer:** {row.get('correct', 'A')}")
                    
                    if row.get('fun_fact'):
                        st.info(f"**Fun Fact:** {row['fun_fact']}")
            
            # Selection management buttons
            col1, col2, col3 = st.columns(3)
            # with col1:
            #     if st.button("✅ Select All Visible", key="select_all_visible"):
            #         # Add all visible quizzes to selection
            #         for idx in display_df.index:
            #             quiz_key = f"quiz_{idx}"
            #             st.session_state.selected_quizzes.add(quiz_key)
            #         st.rerun()
            
            # with col2:
            #     if st.button("❌ Clear Selection", key="clear_selection"):
            #         st.session_state.selected_quizzes.clear()
            #         st.rerun()
            
            with col3:
                st.info(f"Selected: {len(st.session_state.selected_quizzes)} quizzes")
            
            # Export functionality with selection options
            with st.expander("📤 Export Quizzes"):
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # Export selected quizzes
                    if st.session_state.selected_quizzes:
                        # Get the actual indices from selected_quizzes
                        selected_indices_for_export = []
                        for quiz_key in st.session_state.selected_quizzes:
                            if quiz_key.startswith('quiz_'):
                                try:
                                    idx = int(quiz_key.split('_')[1])
                                    # Only include indices that exist in the original quiz_df
                                    if idx in quiz_df.index:
                                        selected_indices_for_export.append(idx)
                                except ValueError:
                                    continue
                        
                        if selected_indices_for_export:
                            selected_df = quiz_df.loc[selected_indices_for_export]
                            formatted_selected_df = self.export_manager.format_for_export(selected_df, "quizzes")
                            csv_data_selected = self.export_manager.get_csv_download_link(formatted_selected_df, "selected_quiz_questions.csv")
                            
                            st.download_button(
                                f"Download Selected Quizzes ({len(selected_df)})",
                                csv_data_selected,
                                "selected_quiz_questions.csv",
                                "text/csv",
                                disabled=st.session_state.generating_section is not None and st.session_state.generating_section != current_tab,
                                help="Export only the quizzes you've manually selected",
                                key="download_selected"
                            )
                        else:
                            st.info("No valid quizzes selected for export")
                    else:
                        st.info("No quizzes selected for export")
                
                with export_col2:
                    # Export all filtered quizzes (original behavior)
                    formatted_df = self.export_manager.format_for_export(display_df, "quizzes")
                    csv_data_all = self.export_manager.get_csv_download_link(formatted_df, "quiz_questions.csv")
                    
                    st.download_button(
                        f"Download All Filtered Quizzes ({len(display_df)})",
                        csv_data_all,
                        "quiz_questions.csv",
                        "text/csv",
                        disabled=st.session_state.generating_section is not None and st.session_state.generating_section != current_tab,
                        help="Export all quizzes currently visible after filtering",
                        key="download_all"
                    )

    def render_polls_tab(self):
        """Render Poll generator tab using cached data"""
        st.header("🗳️ Story Poll Generator")
        
        has_data = (st.session_state.postgres_has_data and 
                st.session_state.insights and 
                st.session_state.insights.get('total_posts', 0) > 0)
        
        if not has_data:
            st.info("👆 Upload your Instagram analytics CSV to get started")
            return
        
        if not self.check_api_key():
            st.warning("⚠️ OpenAI API key required for content generation")
            return

        current_tab = "polls"
        disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
        
        if disabled:
            st.warning(f"⚠️ Controls disabled while `{st.session_state.generating_section}` generation is in progress.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.slider("Poll Batch Size", 5, 30, 20, disabled=disabled)
        
        with col2:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Easy", "Medium", "Hard"],
                index=1,  # Default to Medium
                disabled=disabled,
                help="Easy: Mainstream artists & popular topics\nMedium: Mixed difficulty\nHard: Underground artists & niche topics",
                key="poll_difficulty"
            )
        
        with col3:
            num_options = st.selectbox(
                "Number of Options",
                [2, 4],
                index=0,  # Default to 2 options
                disabled=disabled,
                help="Choose between 2-option (A/B) or 4-option (A/B/C/D) polls",
                key="poll_options"
            )
        
        generate_clicked = st.button("🗳️ Generate Balanced Polls", type="primary", disabled=disabled)
        if generate_clicked:
            st.session_state.generating_section = current_tab
            st.session_state._pending_polls = True
            st.rerun()
        
        if st.session_state._pending_polls and st.session_state.generating_section == current_tab:
            with st.spinner("Creating balanced poll questions..."):
                try:
                    insights = st.session_state.insights or {}
                    
                    # Get user query from session state
                    user_query_for_generation = st.session_state.get('_poll_user_query', None)
                    
                    polls_df = self.content_generator.generate_story_polls(
                        insights,
                        batch_size=batch_size,
                        user_query=user_query_for_generation,
                        difficulty=difficulty.lower(),
                        num_options=num_options,  # Pass the option count
                        chat_history=None
                    )
                    st.session_state.generated_content['polls'] = polls_df
                    
                    # Store extracted examples from user query for potential saving
                    if hasattr(self.content_generator.poll_generator, 'last_extracted_examples'):
                        extracted = self.content_generator.poll_generator.last_extracted_examples
                        if extracted:
                            st.session_state.last_poll_generation_examples = extracted
                    
                    # Show success message with hint about saving examples
                    if user_query_for_generation and user_query_for_generation.strip():
                        st.success(f"✅ Generated {len(polls_df)} {difficulty.lower()} balanced polls using your examples!")
                        if 'last_poll_generation_examples' in st.session_state and st.session_state.last_poll_generation_examples:
                            st.info(f"💡 {len(st.session_state.last_poll_generation_examples)} examples were extracted from your description. You can save them below if they worked well!")
                    else:
                        st.success(f"✅ Generated {len(polls_df)} {difficulty.lower()} balanced polls!")
                except Exception as e:
                    self.show_error_with_delay(f"Poll generation failed: {str(e)}", delay_seconds=10)
                finally:
                    st.session_state._pending_polls = False
                    st.session_state.generating_section = None
                    if '_poll_user_query' in st.session_state:
                        del st.session_state._poll_user_query
                    st.rerun()
        
        if 'polls' in st.session_state.generated_content and st.session_state.generated_content['polls'] is not None:
            polls_df = st.session_state.generated_content['polls']
            
            st.subheader(f"🗳️ Generated {difficulty} Polls ({len(polls_df)} total)")
            
            col1, col2 = st.columns(2)
            with col1:
                balance_disabled = st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
                # Adjust balance threshold based on number of options
                min_balance = 80 if num_options == 4 else 85
                balance_threshold = st.slider("Min Balance Score", 0, 100, min_balance, disabled=balance_disabled)
            with col2:
                score_disabled = balance_disabled
                score_threshold = st.slider("Min Predicted Score", 0, 100, 75, key="poll_score", disabled=score_disabled)
            
            filtered_df = polls_df[
                (polls_df['balance_score'] >= balance_threshold) & 
                (polls_df['predicted_score'] >= score_threshold)
            ]
            
            # Display filtered polls
            for idx, row in filtered_df.iterrows():
                difficulty_badge = f" ({row.get('difficulty', 'medium').upper()})" if 'difficulty' in row else ""
                options_badge = f" | {row.get('num_options', 2)}-option" if 'num_options' in row else ""
                
                with st.expander(f"🗳️ {row.get('theme', 'Poll')}{difficulty_badge}{options_badge} - Balance: {row.get('balance_score', 0):.1f}%"):
                    st.write("**Poll Question:**", row.get('prompt', ''))
                    
                    if row.get('num_options', 2) == 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Option A:** {row.get('option_a', '')}")
                        with col2:
                            st.info(f"**Option B:** {row.get('option_b', '')}")
                    else:
                        # 4-option layout
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Option A:** {row.get('option_a', '')}")
                            st.info(f"**Option C:** {row.get('option_c', '')}")
                        with col2:
                            st.info(f"**Option B:** {row.get('option_b', '')}")
                            st.info(f"**Option D:** {row.get('option_d', '')}")
                    
                    st.write("**Predicted Split:**", row.get('predicted_split', ''))
                    st.write("**Balance Score:**", f"{row.get('balance_score', 0):.1f}%")
            
            # Export functionality - local operation
            with st.expander("📤 Export Polls"):
                formatted_df = self.export_manager.format_for_export(filtered_df, "polls")
                csv_data = self.export_manager.get_csv_download_link(formatted_df, "story_polls.csv")
                st.download_button(
                    "Download Story Polls CSV",
                    csv_data,
                    "story_polls.csv",
                    "text/csv",
                    disabled=st.session_state.generating_section is not None and st.session_state.generating_section != current_tab
                )
    
    def render_chat_tab(self):
        """Render the analytics chat tab"""
        has_data = (st.session_state.postgres_has_data and 
                   st.session_state.insights and 
                   st.session_state.insights.get('total_posts', 0) > 0)
        
        if not has_data:
            st.info("👆 Upload your Instagram analytics CSV to get started")
            return
        
        if not self.check_api_key():
            st.warning("⚠️ OpenAI API key required for chat features")
            return
        
        if self.chat_interface:
            # Ensure RAG pipeline has latest data
            if self.chat_pipeline and st.session_state.insights:
                # Always keep pipeline's insights fresh (cheap assignment)
                self.chat_pipeline.update_analytics_data(None, st.session_state.insights)
            
            # Render the chat interface
            self.chat_interface.render_chat_interface()
        else:
            st.warning("Chat features are temporarily unavailable")
    
    def export_all_data(self):
        """Export all generated data"""
        try:
            export_data = {}
            
            # 1. Export analytics data from PostgreSQL
            if self.db_engine and st.session_state.postgres_has_data:
                try:
                    with self.db_engine.connect() as conn:
                        # Get all analytics data
                        all_data_query = text("""
                            SELECT 
                                post_id,
                                description,
                                publish_time,
                                permalink,
                                post_type,
                                engagements,
                                engagement_rate_impr,
                                follow_conversion,
                                dow,
                                hour_local,
                                time_bucket,
                                content_type_tags,
                                hook_type,
                                main_artists,
                                subgenre,
                                ai_post_score as content_score
                            FROM instagram_data 
                            ORDER BY ai_post_score DESC NULLS LAST, publish_time DESC
                        """)
                        analytics_df = pd.read_sql(all_data_query, conn)
                        
                        if not analytics_df.empty:
                            export_data['analytics'] = self.export_manager.format_for_export(
                                analytics_df, "analytics"
                            )
                except Exception as e:
                    st.error(f"Error exporting analytics data: {str(e)}")
            
            # 2. Export all generated content - with selection support for quizzes
            for content_type, df in st.session_state.generated_content.items():
                if df is not None and not df.empty:
                    try:
                        if content_type == "quizzes" and 'selected_quizzes' in st.session_state and st.session_state.selected_quizzes:
                            # Export only selected quizzes
                            selected_indices = []
                            for quiz_key in st.session_state.selected_quizzes:
                                if quiz_key.startswith('quiz_'):
                                    try:
                                        idx = int(quiz_key.split('_')[1])
                                        selected_indices.append(idx)
                                    except ValueError:
                                        continue
                            
                            if selected_indices:
                                selected_df = df.loc[selected_indices]
                                export_data[f'selected_quizzes'] = self.export_manager.format_for_export(selected_df, content_type)
                        else:
                            # Export all content of this type
                            if content_type in ["reels", "quizzes", "polls"]:
                                export_data[content_type] = self.export_manager.format_for_export(df, content_type)
                            else:
                                export_data[content_type] = df
                    except Exception as e:
                        st.error(f"Error exporting {content_type}: {str(e)}")
            
            # 3. Export artist list if available
            if self.artist_list_manager:
                try:
                    artist_df = self.export_manager.get_artist_list_data(self.db_engine)
                    if not artist_df.empty:
                        export_data['artist_list'] = artist_df
                except Exception as e:
                    st.error(f"Error exporting artist list: {str(e)}")
            
            if export_data:
                # Create timestamp prefix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Export all files
                file_paths = self.export_manager.batch_export_csv(
                    export_data, 
                    prefix=f"music_curator_export_{timestamp}"
                )
                
                # Show success message with download links
                st.success(f"✅ Exported {len(file_paths)} files!")
                
            else:
                st.warning("⚠️ No data available to export")
                
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    
    def render_artist_list_section(self):
        """Render artist list upload section in sidebar"""
        # Artist list upload
        st.header("📤 Upload Artist List")
        if "artist_uploader_key" not in st.session_state:
                st.session_state.artist_uploader_key = 0
        uploaded_artist_file = st.file_uploader(
            "Upload Artist List",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or XLSX file with columns: Artist Name, Genre, Secondary Genre",
            key=f"artist_uploader_{st.session_state.artist_uploader_key}"
        )
        
        if uploaded_artist_file is not None:
            # Validate the file first
            validation_result = self.artist_list_manager.validate_artist_list_file(uploaded_artist_file)
            
            if not validation_result['valid']:
                st.error("❌ Invalid file format:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
                return
            
            # Show warnings if any
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            process_uploaded_files(uploaded_artist_file)
            # Process the file automatically when uploaded
            st.session_state.artist_uploader_key += 1

            # Rerun the app to refresh uploader
            st.rerun()


def main():
    """Main application entry point"""
    # Use cached app instance if available
    if 'app_instance' not in st.session_state:
        st.session_state.app_instance = MusicCuratorApp()
    
    app = st.session_state.app_instance
    app.run()

if __name__ == "__main__":
    main()