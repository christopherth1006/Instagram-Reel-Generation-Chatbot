"""
Export Manager for CSV and Google Sheets functionality
Updated for PostgreSQL and new data structure
"""
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
from typing import Optional, Dict, Any, List
from io import StringIO
import numpy as np
from datetime import datetime
from sqlalchemy import text

class ExportManager:
    def __init__(self, google_credentials_path: Optional[str] = None):
        self.google_credentials_path = google_credentials_path
        self.gc = None
        
        if google_credentials_path and os.path.exists(google_credentials_path):
            self._setup_google_sheets()
    
    def _setup_google_sheets(self):
        """Setup Google Sheets API connection"""
        try:
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.google_credentials_path, scope
            )
            self.gc = gspread.authorize(creds)
        except Exception as e:
            print(f"Warning: Could not setup Google Sheets API: {e}")
            self.gc = None

    def get_analytics_data_from_db(self, db_engine, limit: Optional[int] = None, 
                                 query_type: str = "all") -> pd.DataFrame:
        """
        Get analytics data from PostgreSQL database with flexible query options
        """
        try:
            with db_engine.connect() as conn:
                if query_type == "top_posts":
                    query = text("""
                        SELECT 
                            post_id,
                            account_username,
                            account_name,
                            description,
                            publish_time,
                            permalink,
                            post_type,
                            views,
                            reach,
                            likes,
                            comments,
                            shares,
                            saves,
                            follows,
                            profile_visits,
                            replies,
                            navigation,
                            sticker_taps,
                            link_clicks,
                            engagements,
                            engagement_rate_impr,
                            save_rate,
                            share_rate,
                            follow_conversion,
                            ai_post_score as content_score,
                            content_score as calculated_score,
                            dow,
                            hour_local,
                            time_bucket,
                            content_type_tags,
                            hook_type,
                            main_artists,
                            subgenre,
                            source_file
                        FROM instagram_data 
                        WHERE ai_post_score > 0
                        ORDER BY ai_post_score DESC
                        LIMIT :limit
                    """)
                    params = {"limit": limit or 50}
                    
                elif query_type == "hook_performance":
                    query = text("""
                        SELECT 
                            hook_type,
                            COUNT(*) as post_count,
                            AVG(ai_post_score) as avg_content_score,
                            AVG(engagement_rate_impr) as avg_engagement_rate,
                            AVG(follow_conversion) as avg_follow_conversion,
                            MAX(ai_post_score) as top_content_score,
                            SUM(follows) as total_follows,
                            SUM(engagements) as total_engagements
                        FROM instagram_data 
                        WHERE hook_type IS NOT NULL AND hook_type != ''
                        GROUP BY hook_type
                        ORDER BY avg_content_score DESC
                    """)
                    params = {}
                    
                elif query_type == "time_performance":
                    query = text("""
                        SELECT 
                            time_bucket,
                            COUNT(*) as post_count,
                            AVG(ai_post_score) as avg_content_score,
                            AVG(engagement_rate_impr) as avg_engagement_rate,
                            AVG(follow_conversion) as avg_follow_conversion,
                            MIN(publish_time) as earliest_post,
                            MAX(publish_time) as latest_post
                        FROM instagram_data 
                        WHERE time_bucket IS NOT NULL AND time_bucket != ''
                        GROUP BY time_bucket
                        ORDER BY 
                            CASE time_bucket
                                WHEN 'Morning (6a-12p)' THEN 1
                                WHEN 'Afternoon (12p-6p)' THEN 2
                                WHEN 'Evening (6p-12a)' THEN 3
                                WHEN 'Night (12a-6a)' THEN 4
                                ELSE 5
                            END
                    """)
                    params = {}
                    
                elif query_type == "artist_performance":
                    query = text("""
                        SELECT 
                            main_artists as artist,
                            COUNT(*) as post_count,
                            AVG(ai_post_score) as avg_content_score,
                            AVG(engagement_rate_impr) as avg_engagement_rate,
                            AVG(follow_conversion) as avg_follow_conversion,
                            SUM(follows) as total_follows,
                            SUM(engagements) as total_engagements,
                            MIN(publish_time) as first_post_date,
                            MAX(publish_time) as last_post_date
                        FROM instagram_data 
                        WHERE main_artists IS NOT NULL AND main_artists != ''
                        GROUP BY main_artists
                        HAVING COUNT(*) >= 1
                        ORDER BY avg_content_score DESC
                        LIMIT :limit
                    """)
                    params = {"limit": limit or 100}
                    
                elif query_type == "source_summary":
                    query = text("""
                        SELECT 
                            source_file,
                            COUNT(*) as post_count,
                            AVG(ai_post_score) as avg_content_score,
                            AVG(engagement_rate_impr) as avg_engagement_rate,
                            AVG(follow_conversion) as avg_follow_conversion,
                            MIN(publish_time) as earliest_post,
                            MAX(publish_time) as latest_post,
                            COUNT(DISTINCT main_artists) as unique_artists
                        FROM instagram_data 
                        WHERE source_file IS NOT NULL AND source_file != ''
                        GROUP BY source_file
                        ORDER BY post_count DESC
                    """)
                    params = {}
                    
                else:  # all data
                    query = text("""
                        SELECT 
                            post_id,
                            account_username,
                            account_name,
                            description,
                            publish_time,
                            permalink,
                            post_type,
                            views,
                            reach,
                            likes,
                            comments,
                            shares,
                            saves,
                            follows,
                            profile_visits,
                            replies,
                            navigation,
                            sticker_taps,
                            link_clicks,
                            engagements,
                            engagement_rate_impr,
                            save_rate,
                            share_rate,
                            follow_conversion,
                            ai_post_score as content_score,
                            content_score as calculated_score,
                            dow,
                            hour_local,
                            time_bucket,
                            content_type_tags,
                            hook_type,
                            main_artists,
                            subgenre,
                            source_file,
                            processed_at
                        FROM instagram_data 
                        ORDER BY ai_post_score DESC NULLS LAST, publish_time DESC
                    """)
                    params = {}
                
                if limit and query_type == "all":
                    query = text(str(query) + " LIMIT :limit")
                    params = {"limit": limit}
                
                result = conn.execute(query, params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
                
        except Exception as e:
            print(f"Error getting analytics data from database: {e}")
            return pd.DataFrame()

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV and return the file path"""
        try:
            # Ensure the exports directory exists
            os.makedirs('exports', exist_ok=True)
            
            file_path = f"exports/{filename}"
            # Use UTF-8 encoding with BOM for proper emoji support
            df.to_csv(file_path, index=False, encoding='utf-8-sig', lineterminator='\n')
            return file_path
        except Exception as e:
            raise Exception(f"Failed to export CSV: {e}")
    
    def get_csv_download_link(self, df: pd.DataFrame, filename: str = None) -> str:
        """Generate CSV download link for Streamlit"""
        csv_buffer = StringIO()
        # Use UTF-8 encoding with BOM for proper emoji support
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig', lineterminator='\n')
        return csv_buffer.getvalue()
    
    def export_to_csv_excel_compatible(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV with enhanced Excel compatibility for emojis"""
        try:
            # Ensure the exports directory exists
            os.makedirs('exports', exist_ok=True)
            
            file_path = f"exports/{filename}"
            
            # First, ensure all text columns are properly formatted
            df_formatted = df.copy()
            for col in df_formatted.columns:
                if df_formatted[col].dtype == 'object':
                    # Ensure proper string formatting and handle any encoding issues
                    df_formatted[col] = df_formatted[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            
            # Export with UTF-8 BOM for maximum Excel compatibility
            df_formatted.to_csv(file_path, index=False, encoding='utf-8-sig', lineterminator='\n')
            
            return file_path
        except Exception as e:
            raise Exception(f"Failed to export Excel-compatible CSV: {e}")
    
    def export_to_google_sheets(self, 
                               df: pd.DataFrame, 
                               sheet_name: str,
                               worksheet_name: str = "Sheet1") -> Optional[str]:
        """Export DataFrame to Google Sheets"""
        if not self.gc:
            raise Exception("Google Sheets API not configured")
        
        try:
            # Try to open existing sheet or create new one
            try:
                sheet = self.gc.open(sheet_name)
            except gspread.SpreadsheetNotFound:
                sheet = self.gc.create(sheet_name)
                # Make sheet publicly readable (optional)
                sheet.share('', perm_type='anyone', role='reader')
            
            # Select or create worksheet
            try:
                worksheet = sheet.worksheet(worksheet_name)
                worksheet.clear()  # Clear existing data
            except gspread.WorksheetNotFound:
                worksheet = sheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            
            # Convert DataFrame to list of lists for gspread
            data = [df.columns.tolist()] + df.values.tolist()
            
            # Update the worksheet
            worksheet.update('A1', data)
            
            return sheet.url
        
        except Exception as e:
            raise Exception(f"Failed to export to Google Sheets: {e}")
    
    def create_export_summary(self, export_data: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary of exported data"""
        summary_data = []
        
        for data_type, df in export_data.items():
            if isinstance(df, pd.DataFrame):
                summary_data.append({
                    'Data Type': data_type,
                    'Records': len(df),
                    'Columns': len(df.columns),
                    'Export Time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return pd.DataFrame(summary_data)
    
    def batch_export_csv(self, data_dict: Dict[str, pd.DataFrame], prefix: str = "") -> Dict[str, str]:
        """Export multiple DataFrames to CSV files with Excel-compatible emoji support"""
        file_paths = {}
        
        for name, df in data_dict.items():
            filename = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
            # Use Excel-compatible export for better emoji support
            file_path = self.export_to_csv_excel_compatible(df, filename)
            file_paths[name] = file_path
        
        return file_paths

    def format_for_export(self, df: pd.DataFrame, export_type: str = "general") -> pd.DataFrame:
        """Format DataFrame for specific export types"""
        if df.empty:
            return df
        
        df_formatted = df.copy()
        
        if export_type == "analytics":
            # Select and order columns for final export
            desired_columns = [
                'post_id', 'description',
                'publish_time', 'permalink', 'post_type', 
                'engagements', 'engagement_rate_impr', 'save_rate', 'share_rate', 'follow_conversion',
                'ai_post_score', 'dow', 'hour_local', 'time_bucket', 
                'content_type_tags', 'hook_type', 'main_artists', 'subgenre'
            ]
            
            # Ensure all desired columns exist
            for col in desired_columns:
                if col not in df_formatted.columns:
                    # Set appropriate default values
                    if col in ['engagements', 'engagement_rate_impr', 'save_rate', 'share_rate', 
                              'follow_conversion', 'ai_post_score', 'hour_local']:
                        df_formatted[col] = 0
                    elif col in ['views', 'reach', 'likes', 'comments', 'shares', 'saves', 'follows',
                                'profile_visits', 'replies', 'navigation', 'sticker_taps', 'link_clicks']:
                        df_formatted[col] = 0
                    else:
                        df_formatted[col] = ''
            
            # Reorder columns to match desired structure
            available_columns = [col for col in desired_columns if col in df_formatted.columns]
            df_formatted = df_formatted[available_columns]
        
        elif export_type == "performance":
            # Format performance analysis exports
            if 'avg_content_score' in df_formatted.columns:
                df_formatted['avg_content_score'] = df_formatted['avg_content_score'].round(2)
            if 'avg_engagement_rate' in df_formatted.columns:
                df_formatted['avg_engagement_rate'] = df_formatted['avg_engagement_rate'].round(2)
            if 'avg_follow_conversion' in df_formatted.columns:
                df_formatted['avg_follow_conversion'] = df_formatted['avg_follow_conversion'].round(2)
        
        # Clean up data for export
        for col in df_formatted.columns:
            if df_formatted[col].dtype == 'object':
                def _normalize_cell(x):
                    if isinstance(x, list):
                        return ' | '.join(x)
                    if pd.isna(x):
                        return ''
                    s = str(x)
                    return '' if s.lower() == 'nan' else s
                df_formatted[col] = df_formatted[col].apply(_normalize_cell)
            
            # Format numeric columns
            elif df_formatted[col].dtype in ['float64', 'int64']:
                df_formatted[col] = df_formatted[col].round(4)
        
        return df_formatted

    def export_comprehensive_analytics(self, db_engine, export_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Export comprehensive analytics data including multiple analysis types
        """
        if export_types is None:
            export_types = ["all", "top_posts", "hook_performance", "time_performance", 
                           "artist_performance", "source_summary"]
        
        export_data = {}
        
        for export_type in export_types:
            try:
                df = self.get_analytics_data_from_db(db_engine, query_type=export_type)
                if not df.empty:
                    formatted_df = self.format_for_export(df, 
                                                        "analytics" if export_type in ["all", "top_posts"] else "performance")
                    export_data[export_type] = formatted_df
            except Exception as e:
                print(f"Error exporting {export_type}: {e}")
        
        return export_data

    def generate_analytics_report(self, db_engine) -> pd.DataFrame:
        """
        Generate a comprehensive analytics report with key metrics
        """
        try:
            with db_engine.connect() as conn:
                # Get overall statistics
                stats_query = text("""
                    SELECT 
                        COUNT(*) as total_posts,
                        AVG(ai_post_score) as avg_content_score,
                        MAX(ai_post_score) as max_content_score,
                        AVG(engagement_rate_impr) as avg_engagement_rate,
                        AVG(follow_conversion) as avg_follow_conversion,
                        COUNT(DISTINCT main_artists) as unique_artists,
                        COUNT(DISTINCT source_file) as data_sources,
                        MIN(publish_time) as data_start_date,
                        MAX(publish_time) as data_end_date
                    FROM instagram_data
                    WHERE ai_post_score > 0
                """)
                stats_result = conn.execute(stats_query)
                stats = stats_result.fetchone()
                
                # Get top performing content types
                content_type_query = text("""
                    SELECT 
                        hook_type,
                        COUNT(*) as post_count,
                        AVG(ai_post_score) as avg_score,
                        AVG(engagement_rate_impr) as avg_engagement
                    FROM instagram_data
                    WHERE hook_type IS NOT NULL AND hook_type != ''
                    GROUP BY hook_type
                    ORDER BY avg_score DESC
                    LIMIT 10
                """)
                content_type_df = pd.read_sql(content_type_query, conn)
                
                # Create summary report
                summary_data = {
                    'Metric': [
                        'Total Posts',
                        'Average Content Score',
                        'Maximum Content Score', 
                        'Average Engagement Rate',
                        'Average Follow Conversion Rate',
                        'Unique Artists',
                        'Data Sources',
                        'Data Coverage Start',
                        'Data Coverage End'
                    ],
                    'Value': [
                        stats[0],
                        f"{stats[1]:.2f}" if stats[1] else 'N/A',
                        f"{stats[2]:.2f}" if stats[2] else 'N/A',
                        f"{stats[3]:.2f}%" if stats[3] else 'N/A',
                        f"{stats[4]:.2f}%" if stats[4] else 'N/A',
                        stats[5],
                        stats[6],
                        stats[7].strftime('%Y-%m-%d') if stats[7] else 'N/A',
                        stats[8].strftime('%Y-%m-%d') if stats[8] else 'N/A'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                
                # Add top performing content types
                if not content_type_df.empty:
                    content_type_summary = content_type_df[['hook_type', 'avg_score', 'avg_engagement']].head(5)
                    content_type_summary['avg_score'] = content_type_summary['avg_score'].round(2)
                    content_type_summary['avg_engagement'] = content_type_summary['avg_engagement'].round(2)
                
                return summary_df
                
        except Exception as e:
            print(f"Error generating analytics report: {e}")
            return pd.DataFrame()

    def get_artist_list_data(self, db_engine) -> pd.DataFrame:
        """Get artist list data from database"""
        try:
            with db_engine.connect() as conn:
                query = text("SELECT * FROM artist_list ORDER BY artist_name")
                return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error getting artist list data: {e}")
            return pd.DataFrame()

    def export_artist_list(self, db_engine, filename: str = "artist_list.csv") -> str:
        """Export artist list to CSV"""
        try:
            artist_df = self.get_artist_list_data(db_engine)
            if not artist_df.empty:
                return self.export_to_csv_excel_compatible(artist_df, filename)
            else:
                raise Exception("No artist data available for export")
        except Exception as e:
            raise Exception(f"Failed to export artist list: {e}")