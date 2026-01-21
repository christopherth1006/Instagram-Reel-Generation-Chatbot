# langchain_chat_workflow.py
from typing import Annotated, Dict, Any, List, TypedDict
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.function import FunctionMessage
import streamlit as st
import re
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from utils.artist_list_manager import ArtistListManager
from utils.process_upload import get_engine
from generators.content_generator import ContentGenerator
import json


class SQLQueryModel(BaseModel):
    sql_query: Annotated[str, "The SQL query string generated based on the user's input. This field mustn't contain other string except the exact SQL query."]
    table_name: Annotated[Literal["instagram_data", "artist_data", "demographics_followers_age_gender", "demographics_followers_top_cities", "demographics_followers_top_countries", "followers_metric_data"], ""] = "None"
    column_name: Annotated[Literal[
        "post_id", "account_id", "account_username", "account_name", "description",
        "duration_(sec)", "publish_time", "permalink", "post_type", "views",
        "reach", "likes", "shares", "follows", "comments", "saves", "profile_visits",
        "replies", "navigation", "sticker_taps", "link_clicks", "source_file",
        "engagements", "engagement_rate_impr", "save_rate", "share_rate",
        "follow_conversion", "ai_post_score", "dow", "hour_local", "time_bucket",
        "content_type_tags", "hook_type", "main_artists", "subgenre", "artist_name",
        "genre", "secondary_genre", "age_range", "men", "women", "city", 
        "followers_percent", "country", "date", "followers_count"
    ], ""] = "None"

class ReelGeneratorModel(BaseModel):
    batch_size: Annotated[int, ""]
    focus_themes: Annotated[Literal["treding artist", "classic hits", "new releases", "underground", "throwback"], ""] = "trending artist"

class QuizGeneratorModel(BaseModel):
    batch_size: Annotated[int, ""]
    quiz_type: Annotated[Literal["who said it", "fill in the blank", "guess the year", "sample match"], ""] = "who said it"
    difficulty: Annotated[Literal["easy", "medium", "hard"], ""] = "medium"

class PollGeneratorModel(BaseModel):
    batch_size: Annotated[int, ""]
    difficulty: Annotated[Literal["easy", "medium", "hard"], ""] = "medium"
    num_options: Annotated[int, "The number of options for each poll."] = 2

# ------------------------------
class PostgresChatWorkflow:
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        self.db_engine = get_engine()
        self.artist_list_manager = ArtistListManager(db_engine=self.db_engine)
        self.content_generator = ContentGenerator(
            OPENAI_API_KEY, 
            self.artist_list_manager,
            self.db_engine
        )

    def _get_system_prompt(self) -> str:
        """System prompt for intent analysis and final response generation."""
        return f"""
        You are an intelligent and helpful chatbot. 
        Your role is to answer user questions and perform tasks using data and AI models.

        You have access to tools that allow you to query a database and generate creative content. 
        Always decide which tool to use based on the user’s intent.

        ──────────────────────────────
        TOOL USAGE RULES
        ──────────────────────────────
        1. **Database Querying (SQLQueryModel)**
        - **PRIMARY DIRECTIVE:** **ALWAYS** use this tool if the user's question involves analytics, metrics, data retrieval, or historical performance. This includes specific data points like a followers count on a given date.
        - Call the tool with a **single SQL query string only**, no explanations or other text.
        - After receiving the query result, use it to generate a clear, human-friendly answer.
        - **STRICT TABLE MAPPING:** When generating SQL, strictly use the relevant tables based on the category of the question. **DO NOT** use generic, unlisted table names

        Analytics-related questions include:
        - Post performance (engagements, likes, reach, views, saves, etc.)
        - Top or best-performing posts/stories
        - Publish time per post analysis
        - Time-based trends (by hour, day, week)
        - Comparisons (artists, hook types, post types)
        - Follower growth and conversion effectiveness
        - Content scores, engagement rates, or averages/totals
        - **Follwer Demographics data (Age, Gender, City, Country)**, which requires using the `demographics_followers_age_gender`, `demographics_followers_top_cities`, and `demographics_followers_top_countries` tables. Joins across these tables may be necessary.
        - **Follower Count Growth and Analytics Over Date"" which requires using the `followers_metric_data` table and their own `followers_count` and `date` columns.

        **IMPORTANT POST-QUERY INSTRUCTION (Handling Missing Data):**
        - If the user asks for a specific metric on a **future date** or a date that is **not found** in the database after the SQL query:
            1. **DO NOT** make up or hallucinate the data.
            2. State clearly that the data for the requested date is not available.
            3. Then, and **only then**, suggest a projection or trend analysis. *Do not bypass the SQL call to make a projection.*
            
        2. **Reel Generation (ReelGeneratorModel)**
        - Use when the user asks to create or update Instagram reels.
        - Input parameters:
            • batch_size → default: 5  
            • focus_themes → default: "trending artist"
        - Always include both parameters when calling the tool. 
        If the user doesn’t specify them, use the defaults above.
        - Call the tool with only the batch size and focus themes, no other text.
        - After receiving results, summarize them into an engaging, natural response
        as if you’re talking to a creator or marketer. Use emojis, tone, and narrative flow
        appropriate for Instagram reels content.

        💡 **Chat Awareness of Captions & Hashtags**
        - The “description” field in the `instagram_data` table contains both **captions and hashtags**.
        - When generating new reel or post ideas, always analyze and reference the language, tone,
        keywords, and hashtags found in this field.
        - Use them to understand the creator’s voice, content style, and engagement trends.
        - Example:
        description: "Comment 'Vibe20' if you want the link to the posters🔥 #raptok #music #reels #reelsinstagram #rnb #rap #rnbmusic #rnbsoul"
        → The AI should recognize this as an engaging caption with call-to-action + relevant music hashtags,
            and use similar phrasing or tags when suggesting new ideas.

        3. **Quiz Generation (QuizGeneratorModel)**
        - Use when the user asks to create or update Instagram quizzes.
        - Input parameters:
            • batch_size → default: 5  
            • quiz_type → default: "who said it"  
            • difficulty → default: "medium"
        - Always include both parameters when calling the tool. 
        If the user doesn’t specify them, use the defaults above.
        - Call the tool with batch size, quiz type, and difficulty, and no other string.
        - After receiving results, summarize them in a conversational, interactive style
        (e.g. “Here are some fun quizzes your audience will love!”).

        4. **Poll Generation (PollGeneratorModel)**
        - Use when the user asks to create or update Instagram polls.
        - Input parameters:
            • batch_size → default: 5  
            • difficulty → default: "medium"
            • num_options → default: 2
        - Always include both parameters when calling the tool. 
        If the user doesn’t specify them, use the defaults above.
        - Call the tool with batch size, difficulty, and num_options only.
        - After receiving polls, summarize them naturally for the user with clear formatting,
        without showing raw data.

        ──────────────────────────────
        DEFAULT VALUES (MUST APPLY)
        ──────────────────────────────
        - batch_size = 5  
        - focus_themes = "trending artist"  
        - quiz_type = "who said it"  
        - difficulty = "medium"

        ──────────────────────────────
        DATABASE SCHEMA
        ──────────────────────────────
        1.
        Table Name:
        **instagram_data**
        
        Contains post performance metrics.

        Columns:
        post_id, account_id, account_username, account_name, description (contains captions + hashtags),
        duration_(sec), publish_time, permalink, post_type, views, reach, likes, shares, follows, comments,
        saves, profile_visits, replies, navigation, sticker_taps, link_clicks, source_file,
        engagements, engagement_rate_impr, save_rate, share_rate, follow_conversion,
        ai_post_score, dow, hour_local, time_bucket, content_type_tags, hook_type,
        main_artists, subgenre.

        2.
        Table Name:
        **artist_data**

        Contains artist information for content generation.

        Columns:
        artist_name, genre, secondary_genre.

        3.
        Table Name:
        **demographics_followers_age_gender**
        
        Contains follower distribution by gender and age group.

        Columns:
        age_range, men, women.

        4. 
        Table Name:
        **demographics_followers_top_cities**

        Contains the top cities where followers are located.

        Columns:
        city, followers_percent.

        5. 
        Table Name:
        **demographics_followers_top_countries**

        Contains the top countries where followers are located.

        Columns:
        country, followers_percent.

        6. 
        Table Name:
        **followers_metric_data**
        Contains follower count per date.

        Columns:
        date, followers_count, source_file.

        ──────────────────────────────
        FINAL RESPONSE INSTRUCTION
        ──────────────────────────────
        After any tool result is returned (SQL query result, generated reels, quizzes, or polls):
        - Interpret the tool output as structured data.
        - Rephrase and summarize it in a natural, engaging, user-friendly format.
        - Include emotional tone, clarity, and brevity (e.g., for Instagram creators).
        - When generating ideas, reference relevant captions and hashtags from the 'description' field
        to maintain stylistic consistency with the user's existing content.
        - Do NOT show raw tables or code.
        - If the user continues with follow-up requests, keep prior conversation context.
        - Always respond as if speaking directly to the user.
        """
    
    def execute_sql_callable(self, sql_query: str, table_name: str = None, column_name: str = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as a list of dictionaries.

        Automatically casts timestamp/date columns in the SELECT clause to TEXT to avoid
        serialization issues, without affecting ORDER BY or other clauses.

        ──────────────────────────────
        DATABASE SCHEMA
        ──────────────────────────────
        1.
        Table Name:
        **instagram_data**
        
        Contains post performance metrics.

        Columns:
        post_id, account_id, account_username, account_name, description (contains captions + hashtags),
        duration_(sec), publish_time, permalink, post_type, views, reach, likes, shares, follows, comments,
        saves, profile_visits, replies, navigation, sticker_taps, link_clicks, source_file,
        engagements, engagement_rate_impr, save_rate, share_rate, follow_conversion,
        ai_post_score, dow, hour_local, time_bucket, content_type_tags, hook_type,
        main_artists, subgenre.

        2.
        Table Name:
        **artist_data**

        Contains artist information for content generation.

        Columns:
        artist_name, genre, secondary_genre.

        3.
        Table Name:
        **demographics_followers_age_gender**
        
        Contains follower distribution by gender and age group.

        Columns:
        age_range, men, women.

        4. 
        Table Name:
        **demographics_followers_top_cities**

        Contains the top cities where followers are located.

        Columns:
        city, followers_percent.

        5. 
        Table Name:
        **demographics_followers_top_countries**

        Contains the top countries where followers are located.

        Columns:
        country, followers_percent.

        6. 
        Table Name:
        **followers_metric_data**
        Contains follower count per date.

        Columns:
        date, followers_count, source_file.

        WARNING: Do not pick the table name other than above tables
        """
        try:
            print(f"sql query (before cast): {sql_query}")

            TABLE_COLUMN_TYPES = {
                "instagram_data": {
                    "publish_time": "timestamp",
                    "created_at": "timestamp",
                    "updated_at": "timestamp",
                    "likes": "int",
                    "engagements": "int",
                    "views": "int"
                },
                "followers_metric_data": {
                    "date": "date",
                    "followers_count": "int"
                },
            }

            if table_name and table_name in TABLE_COLUMN_TYPES:
                match = re.match(r'(SELECT\s+)(.*?)(\s+FROM\s+.*)', sql_query, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    select_clause, columns_str, from_clause = match.groups()
                    columns = [c.strip() for c in columns_str.split(',')]
                    new_columns = []
                    for col in columns:
                        col_name = col.split()[0]
                        col_type = TABLE_COLUMN_TYPES[table_name].get(col_name)
                        if col_type in ["timestamp", "date"]:
                            new_columns.append(f"CAST({col_name} AS TEXT) AS {col_name}")
                        else:
                            new_columns.append(col)
                    sql_query = select_clause + ', '.join(new_columns) + from_clause

            print(f"sql query (after cast): {sql_query}")

            with self.db_engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)

            # ✅ Correct conversion
            df = df.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

            return df.to_dict(orient="records")

        except Exception as e:
            print(f"❌ Error executing SQL: {e}")
            return [{"error": str(e)}]

    def run(self, user_query: str, conversation_history: List = []) -> dict:

        print("bbb")

        self.llm_with_tools = self.llm.bind_tools([SQLQueryModel, ReelGeneratorModel, QuizGeneratorModel, PollGeneratorModel])
        print(f"conversation history: {conversation_history}")
        response = self.llm_with_tools.invoke(conversation_history)
        print(f"all tool: {response.tool_calls}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"tool call: {tool_name} : {tool_args}")
                if tool_name == "SQLQueryModel":
                    print(f"🔧 Tool Call Detected: {tool_name}")
                    print(f"🧩 Args: {tool_args}")

                    # Execute the SQL
                    sql_query = tool_args["sql_query"]

                    # Optional: extract table name from query
                    table_name_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                    table_name = table_name_match.group(1) if table_name_match else None

                    column_name_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
                    column_name = column_name_match.group(1) if column_name_match else None

                    sql_result = self.execute_sql_callable(sql_query, table_name=table_name, column_name=column_name)
                    # sql_result = self.execute_sql_callable(tool_args["sql_query"])
                    
                    result_strings = [json.dumps(item, ensure_ascii=False) for item in sql_result]

                    print("✅ SQL Result:", result_strings)
                    print("type:", type(result_strings))
                    conversation_history.append(
                        FunctionMessage(
                            name="query_results",
                            content="\n".join(result_strings)
                        )
                    )
                    
                
                elif tool_name == "ReelGeneratorModel":
                    print(f"🔧 Tool Call Detected: {tool_name}")
                    print(f"🧩 Args: {tool_args}")
                    batch_size = tool_args.get("batch_size", 5)
                    focus_themes = tool_args.get("focus_themes", "trending artist")
                    
                    reels_df = self.content_generator.generate_reel_scripts(
                        None,
                        batch_size=batch_size,
                        focus_themes=focus_themes,
                        analytics_data=None,
                        chat_history = conversation_history
                    )
                    print(f"reels result: {reels_df}")
                    reel_str_list = reels_df.astype(str).apply(lambda row: ', '.join(row), axis=1).tolist()
                    print("type:", type(reel_str_list))
                    conversation_history.append(
                        FunctionMessage(
                            name="reel_results",
                            content="\n".join(reel_str_list)
                        )
                    )
                    # final_answer = self.llm_with_tools.invoke(conversation_history)

                    # print(f"final answer: {final_answer}")
                    # return {"answer": final_answer.content, "data": reels_df}
                elif tool_name == "QuizGeneratorModel":
                    print(f"🔧 Tool Call Detected: {tool_name}")
                    print(f"🧩 Args: {tool_args}")
                    batch_size = tool_args.get("batch_size", 5)
                    quiz_types = tool_args.get("quiz_type", "who said it")
                    difficulty = tool_args.get("difficulty", "medium")
                    

                    quiz_df = self.content_generator.generate_story_quizzes(
                        None,
                        batch_size=batch_size,
                        quiz_types=quiz_types,
                        difficulty=difficulty.lower(),
                        chat_history = conversation_history
                    )
                    print(f"quizzes result: {quiz_df}")

                    quiz_str_list = quiz_df.astype(str).apply(lambda row: ', '.join(row), axis=1).tolist()
                    print("type:", type(quiz_str_list))
                    conversation_history.append(
                        FunctionMessage(
                            name="quiz_results",
                            content="\n".join(quiz_str_list)
                        )
                    )
                    # final_answer = self.llm_with_tools.invoke(conversation_history)
                    # print(f"final answer: {final_answer}")

                    # return {"answer": final_answer.content, "data": quiz_df}

                elif tool_name == "PollGeneratorModel":
                    print(f"🔧 Tool Call Detected: {tool_name}")
                    print(f"🧩 Args: {tool_args}")
                    batch_size = tool_args.get("batch_size", 5)
                    difficulty = tool_args.get("difficulty", "medium")
                    num_options = tool_args.get("num_options", 2)

                    polls_df = self.content_generator.generate_story_polls(
                        None,
                        batch_size=batch_size,
                        user_query=user_query,
                        difficulty=difficulty.lower(),
                        num_options=num_options,
                        chat_history = conversation_history
                    )
                    print(f"polls result: {polls_df}")
                    poll_str_list = polls_df.astype(str).apply(lambda row: ', '.join(row), axis=1).tolist()
                    conversation_history.append(
                        FunctionMessage(
                            name="poll_results",
                            content="\n".join(poll_str_list)
                        )
                    )
                    # final_answer = self.llm_with_tools.invoke(conversation_history)
                    # print(f"final answer: {final_answer}")

                    # return {"answer": final_answer.content, "data": polls_df}
            final_answer = self.llm.invoke(conversation_history, config={"tool_choice": "none"})
            
            print(f"final answer: {final_answer}")

            return {"answer": final_answer.content}
            

        # If no tool call, return LLM’s direct response
        return {"answer": response.content}

