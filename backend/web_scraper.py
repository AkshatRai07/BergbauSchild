import requests
from bs4 import BeautifulSoup
import feedparser
import os
import json
from urllib.parse import urljoin
from datetime import datetime

# --- Database Imports ---
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, String, MetaData, Text, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import INSERT

# --- AI & Config Imports ---
from dotenv import load_dotenv
import google.generativeai as genai

# --- CONFIGURATION ---
load_dotenv()

# --- Scraper Config ---
DGMS_URL = "https://dgms.gov.in/"
NEWS_RSS_URL = "https://news.google.com/rss/search?q=(%22mining%20accident%22%20OR%20%22coal%20mine%22)%20india&hl=en-IN&gl=IN&ceid=IN:en"

# --- Database Config ---
# Render/Heroku will provide this environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("Warning: DATABASE_URL not set. Using local SQLite for demo.")
    DATABASE_URL = "sqlite:///./scraper.db" # Fallback for local testing

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# --- AI Config ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initialize the Gemini Model
try:
    llm = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest", # Use Flash for speed/cost
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print("Gemini 1.5 Flash model initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini model: {e}")
    llm = None

# --- DATABASE DEFINITION ---
# This table stores ALL incidents, from scraper or user
# Your app.py can READ from this table
incidents_table = Table(
    'incidents',
    metadata,
    Column('id', sqlalchemy.Integer, primary_key=True, autoincrement=True),
    Column('url', String, unique=True, nullable=False), # URL is the unique key
    Column('title', String, nullable=False),
    Column('source', String),
    Column('seen_at', TIMESTAMP, server_default=func.now()),
    
    # --- AI PROCESSED FIELDS ---
    # The AI will fill these fields.
    Column('status', String, default='pending'), # 'pending', 'processed', 'error'
    Column('ai_summary', Text),
    Column('ai_category', String), # e.g., "Incident Report", "New Regulation", "General News"
    Column('ai_code', String)      # e.g., "1.1" (from your codes.json logic)
)

# --- DATABASE FUNCTIONS ---

def init_db():
    """Initializes the database tables."""
    print("Initializing database tables...")
    metadata.create_all(engine)

def log_incident(conn, url: str, title: str, source: str, ai_result: dict):
    """
    Logs a fully processed incident to the database.
    Uses ON CONFLICT DO NOTHING to avoid duplicates based on the 'url' constraint.
    """
    insert_stmt = INSERT(incidents_table).values(
        url=url,
        title=title,
        source=source,
        status=ai_result.get("status", "error"),
        ai_summary=ai_result.get("summary"),
        ai_category=ai_result.get("category"),
        ai_code=ai_result.get("code")
    )
    
    # This ensures we don't crash if we scrape the same link twice
    # It will only insert if the URL is not already in the table
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(
        index_elements=['url']
    )
    
    conn.execute(on_conflict_stmt)

# --- AI FUNCTION ---

AI_SYSTEM_PROMPT = """
You are an autonomous AI agent working for the Directorate General of Mines Safety (DGMS), India.
Your job is to read snippets of text from news articles, circulars, and reports.
For each snippet, you must:
1.  **Summarize** the snippet concisely (2-3 sentences).
2.  **Categorize** the snippet into one: "Incident Report", "Safety Regulation", "Legal/Compliance", "General News", "Other".
3.  **Assign** an accident code (e.g., "1.1", "2.3") IF it is an "Incident Report" and there's enough detail. If not, set code to "N/A".
4.  **Respond ONLY in JSON format.**

Example Input:
"Title: Roof fall at GDK-5 mine; 2 trapped.
Snippet: A preliminary report on the GDK-5 incline incident in Telangana cites violations in roof support as per Regulation 108..."

Example Output:
{
  "summary": "A roof fall incident occurred at the GDK-5 incline mine in Telangana, trapping two workers. The cause is linked to violations in roof support (Regulation 108).",
  "category": "Incident Report",
  "code": "1.1" 
}

Example Input:
"Title: DGMS issues new slope stability guidelines.
Snippet: DGMS issues new comprehensive guidelines for ensuring slope stability in all opencast workings, effective 01/12/2025."

Example Output:
{
  "summary": "New comprehensive guidelines for slope stability in opencast mines have been issued by DGMS, effective December 1, 2025.",
  "category": "Safety Regulation",
  "code": "N/A"
}
"""

def analyze_text_with_gemini(text_snippet: str, title: str) -> dict:
    """
    Sends a text snippet to Gemini for analysis and returns a structured dict.
    """
    if not llm:
        print("AI model not available, skipping analysis.")
        return {"status": "error", "summary": "AI model offline."}

    prompt = f"""
    Title: {title}
    Snippet: {text_snippet}
    """
    
    try:
        response = llm.generate_content([AI_SYSTEM_PROMPT, prompt])
        
        # Clean the response and parse JSON
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        ai_data = json.loads(json_text)
        
        ai_data["status"] = "processed"
        return ai_data

    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return {
            "status": "error",
            "summary": f"Failed to analyze snippet: {e}",
            "category": "Other",
            "code": "N/A"
        }

# --- SCRAPER FUNCTIONS (Now with AI) ---

def scrape_dgms(conn):
    """
    Scrapes DGMS, analyzes new items with AI, and logs to DB.
    """
    print("Checking DGMS for updates...")
    processed_count = 0
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(urljoin(DGMS_URL, "/whats-new"), headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Find the 'What's New' list (this selector is a guess)
        for item in soup.select(".whatsnew-list li a"): 
            title = item.text.strip()
            relative_url = item.get('href')
            if not title or not relative_url:
                continue
            
            absolute_url = urljoin(DGMS_URL, relative_url)
            
            # --- AI PROCESSING STEP ---
            # We use the title as the "content" since that's all we have
            print(f"  [DGMS] Analyzing: {title[:50]}...")
            ai_result = analyze_text_with_gemini(text_snippet=title, title=title)
            
            # Log to DB (the function handles duplicates)
            log_incident(conn, absolute_url, title, "DGMS", ai_result)
            processed_count += 1

    except Exception as e:
        print(f"Error scraping DGMS: {e}")
    
    print(f"DGMS scrape complete. Analyzed {processed_count} items.")

def scrape_google_news(conn):
    """
    Scrapes Google News RSS, analyzes new items with AI, and logs to DB.
    """
    print("Checking Google News RSS for updates...")
    processed_count = 0
    try:
        feed = feedparser.parse(NEWS_RSS_URL)
        
        for entry in feed.entries:
            title = entry.title
            link = entry.link
            # Use the RSS summary as our content snippet
            content_snippet = entry.summary 
            
            # --- AI PROCESSING STEP ---
            print(f"  [News] Analyzing: {title[:50]}...")
            ai_result = analyze_text_with_gemini(text_snippet=content_snippet, title=title)
            
            # Log to DB (the function handles duplicates)
            log_incident(conn, link, title, "Google News", ai_result)
            processed_count += 1
            
    except Exception as e:
        print(f"Error scraping Google News RSS: {e}")
    
    print(f"Google News scrape complete. Analyzed {processed_count} items.")


# --- MAIN JOB FUNCTION ---

def check_for_updates():
    """
    The main function for the Cron Job.
    It scrapes sources, analyzes them with AI, and writes to the database.
    """
    print(f"\n--- [{datetime.now()}] Running autonomous check for updates ---")
    
    # Establish a single connection for this job run
    with engine.connect() as conn:
        # Run all scrapes within a transaction
        with conn.begin():
            scrape_dgms(conn)
            scrape_google_news(conn)
    
    print("--- Autonomous check finished ---")

# --- Direct execution (for testing and for the cron job) ---
if __name__ == "__main__":
    print("Running scraper job...")
    # Init the DB (safe to run, 'CREATE TABLE IF NOT EXISTS')
    init_db()
    # Run the check
    check_for_updates()
    print("Scraper job finished.")