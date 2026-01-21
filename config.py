"""
Configuration settings for the AI Music Curator Agent
"""
import os
from pathlib import Path
import yaml
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent
BRAND_VOICE_PATH = PROJECT_ROOT / "brand_voice.yaml"
IG_EXPORTS_PATH = PROJECT_ROOT / "IG Exports"
TEMPLATE_PATH = PROJECT_ROOT / "Template for Clean IG Ready Analytics.xlsx"

CONTENT_SCORE_WEIGHTS = {
    "follows_per_impression": 0.4,
    "engagement_rate": 0.55,
    "completion_rate": 0.05
}

REEL_BATCH_SIZE = 15
QUIZ_BATCH_SIZE = 50
POLL_BATCH_SIZE = 20

MIN_IMPRESSIONS_THRESHOLD = 100
SMOOTHING_FACTOR = 0.1

OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.6

def load_brand_voice() -> Dict[str, Any]:
    """Load brand voice configuration from YAML"""
    try:
        with open(BRAND_VOICE_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load brand voice config: {e}")
        return {}

BRAND_VOICE = load_brand_voice()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "music-curator-analytics")

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K_RESULTS = 5
RAG_SIMILARITY_THRESHOLD = 0.7

STREAMLIT_CONFIG = {
    "page_title": "AI Music Curator Assistant",
    "page_icon": "🎵",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

