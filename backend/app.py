from dotenv import load_dotenv
import os
import json
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as LangchainBaseModel, Field
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_classic.tools import StructuredTool
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

load_dotenv()
warnings.filterwarnings('ignore')

CSV_PATH = "mining_accidents.csv"
CODES_JSON_PATH = "mining_accidents_codes.json"
UPDATES_JSON_PATH = "updates.json"
VECTOR_INDEX_PATH = "mining_safety_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if not (os.path.exists(CSV_PATH) and os.path.exists(CODES_JSON_PATH)):
    print(f"FATAL ERROR: Missing data files.")
    print(f"Please ensure '{CSV_PATH}' and '{CODES_JSON_PATH}' are in the same directory.")

class MiningAccidentDatabase:
    """Manages the mining accident database and metadata"""
    
    def __init__(self, csv_path: str, codes_json_path: str):
        """Initialize database from CSV and codes JSON"""
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            self.df = pd.DataFrame()
            
        try:
            with open(codes_json_path, 'r') as f:
                self.accident_codes = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {codes_json_path}")
            self.accident_codes = {}

        if not self.df.empty:
            self.df['date_parsed'] = pd.to_datetime(self.df['date'], format='%d/%m/%y', errors='coerce')
            self.df['year'] = self.df['date_parsed'].dt.year
            self.df['month'] = self.df['date_parsed'].dt.month
            print(f"Loaded {len(self.df)} accident records")
        else:
            print("Database initialized with empty DataFrame.")
            
        print(f"Loaded {len(self.accident_codes)} accident codes")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        if self.df.empty:
            return {'error': 'No data loaded'}
        return {
            'total_accidents': len(self.df),
            'total_deaths': int(self.df['deaths_count'].sum()),
            'date_range': f"{self.df['date_parsed'].min()} to {self.df['date_parsed'].max()}",
            'by_category': self.df['accident_category'].value_counts().to_dict(),
            'by_state': self.df['state'].value_counts().to_dict(),
            'by_mine_type': self.df['mine_type'].value_counts().to_dict(),
            'by_year': self.df.groupby('year')['deaths_count'].sum().to_dict()
        }
    
    def search_by_criteria(self,  
                           state: Optional[str] = None,
                           mine_type: Optional[str] = None,
                           year: Optional[int] = None,
                           category: Optional[str] = None) -> pd.DataFrame:
        """Search accidents by various criteria"""
        if self.df.empty:
            return pd.DataFrame()
            
        result = self.df.copy()
        
        if state:
            result = result[result['state'].str.contains(state, case=False, na=False)]
        if mine_type:
            result = result[result['mine_type'].str.contains(mine_type, case=False, na=False)]
        if year:
            result = result[result['year'] == year]
        if category:
            result = result[result['accident_category'].str.contains(category, case=False, na=False)]
        
        return result
    
    def get_code_description(self, code: str) -> str:
        """Get description for an accident code"""
        return self.accident_codes.get(code, "Unknown code")

class MiningRAGSystem:
    """RAG system for mining accident analysis"""
    
    def __init__(self, database: MiningAccidentDatabase, index_path: str):
        """Initialize RAG system"""
        self.db = database
        self.index_path = index_path
        self.vectorstore = None
        self.embeddings = None
        
        print("Initializing RAG system...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if os.path.exists(index_path):
            self.load_index(index_path)
        else:
            print("No existing index found.")
            self._build_vectorstore()
            self.save_index(index_path)
    
    def _build_vectorstore(self):
        """Build vector store from accident records"""
        if self.db.df.empty:
            print("Empty database. RAG vectorstore not built.")
            return
        
        print("Building new vectorstore... This may take a moment.")
        def clean(val) -> str:
            return str(val) if pd.notna(val) and str(val).strip() not in ('', 'nan') else "Not documented"

        documents = []
        for idx, row in self.db.df.iterrows():
            code_desc = self.db.get_code_description(str(row['accident_code']))

            content = f"""Accident ID: {row['accident_id']}
Date: {row['date']}
Mine: {row['mine_name']}
Owner: {row['owner']}
Location: {row['district']}, {row['state']}
Mine Type: {row['mine_type']}
Deaths: {row['deaths_count']}
Accident Code: {row['accident_code']} - {code_desc}
Category: {row['accident_category']}
Incident Description: {clean(row['incident_description'])}
Root Cause Analysis: {clean(row['root_cause'])}
Regulations Violated: {clean(row['regulations_violated'])}
"""
            
            metadata = {
                'accident_id': row['accident_id'],
                'date': row['date'],
                'state': str(row['state']),
                'mine_type': str(row['mine_type']),
                'accident_code': str(row['accident_code']),
                'category': str(row['accident_category']),
                'deaths': int(row['deaths_count'])
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        print("Embedding documents and building FAISS index...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("RAG system ready!")
    
    def save_index(self, path: str):
        """Save the vector index"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load a saved vector index"""
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Index loaded from {path}")
        except Exception as e:
            print(f"Error loading index: {e}. Rebuilding...")
            self._build_vectorstore()
            self.save_index(path)
            
    def get_retriever(self):
        """Get a LangChain retriever object"""
        if self.vectorstore:
            return self.vectorstore.as_retriever(search_kwargs={"k": 4})
        return None

class AccidentClassifier:
    """Classifies new accident descriptions"""
    
    def __init__(self, database: MiningAccidentDatabase):
        self.db = database
        self.pipeline = None

    def train(self):
        """Train the classifier on the existing data"""
        if self.db.df.empty:
            print("Database not available for classifier training.")
            return
            
        print("Training accident classifier...")
        
        df_train = self.db.df.copy()
        df_train = df_train[['incident_description', 'accident_code']].dropna()
        df_train['incident_description'] = df_train['incident_description'].astype(str)
        df_train['accident_code'] = df_train['accident_code'].astype(str)

        if len(df_train) < 10:
            print("Warning: Not enough training data for classifier.")
            return

        X = df_train['incident_description']
        y = df_train['accident_code']
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
            ('clf', LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced'))
        ])
        
        try:
            self.pipeline.fit(X, y)
            print("Classifier trained successfully.")
        except Exception as e:
            print(f"Error during classifier training: {e}")
            self.pipeline = None

    def predict(self, description: str) -> Dict[str, str]:
        """Predict the accident code for a new description"""
        if not self.pipeline:
            return {
                "error": "Classifier not trained or available.",
                "predicted_code": "N/A",
                "code_description": "N/A"
            }
            
        predicted_code = self.pipeline.predict([description])[0]
        code_description = self.db.get_code_description(predicted_code)
        
        return {
            "description": description,
            "predicted_code": predicted_code,
            "code_description": code_description
        }

class ReportGenerator:
    
    def __init__(self, database: MiningAccidentDatabase):
        self.db = database

    def create_safety_report(self, year: Optional[int] = None, state: Optional[str] = None) -> str:
        """Generate automated safety audit report"""
        
        data = self.db.search_by_criteria(year=year, state=state)
        
        if data.empty:
            return "SAFETY AUDIT REPORT - NO DATA FOUND FOR CRITERIA"

        report = f"""
{'='*80}
MINING SAFETY AUDIT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

SCOPE:
Year: {year if year else 'All years'}
State: {state if state else 'All states'}
Records Analyzed: {len(data)}

EXECUTIVE SUMMARY:
- Total Accidents: {len(data)}
- Total Fatalities: {data['deaths_count'].sum()}
- Average Deaths per Accident: {data['deaths_count'].mean():.2f}

HIGH-RISK CATEGORIES:
"""
        top_categories = data.groupby('accident_category')['deaths_count'].sum().nlargest(5)
        for cat, deaths in top_categories.items():
            report += f"  • {cat}: {deaths} deaths\n"
        
        report += "\nGEOGRAPHIC ANALYSIS:\n"
        top_states = data.groupby('state')['deaths_count'].sum().nlargest(5)
        for state_name, deaths in top_states.items():
            report += f"  • {state_name}: {deaths} deaths\n"
        
        report += "\nMINE TYPE BREAKDOWN:\n"
        mine_types = data.groupby('mine_type')['deaths_count'].sum()
        for mtype, deaths in mine_types.items():
            if mtype and isinstance(mtype, str):
                report += f"  • {mtype.capitalize()}: {deaths} deaths\n"
        
        report += f"""
RECOMMENDATIONS:
1. Focus immediate attention on {top_categories.index[0] if len(top_categories) > 0 else 'N/A'} prevention
2. Conduct targeted safety audits in {top_states.index[0] if len(top_states) > 0 else 'N/A'}
3. Enhance training programs for high-risk operations

{'='*80}
END OF REPORT
{'='*80}
"""
        return report
    
class TrendMonitor:
    """Detects anomalies and flagged rising trends from accident data"""

    def __init__(self, database: MiningAccidentDatabase):
        self.db = database

    def detect_anomalies(self) -> list:
        if self.db.df.empty:
            return []

        df = self.db.df.copy()
        alerts = []
        total = len(df)

        category_counts = df['accident_category'].value_counts()
        mean_count = float(category_counts.mean())

        for category, count in category_counts.items():
            if count > mean_count * 2:
                severity = "HIGH" if count > mean_count * 3 else "MEDIUM"
                alerts.append({
                    "type": "category_spike",
                    "severity": severity,
                    "title": f"Elevated rate: {category}",
                    "category": category,
                    "count": int(count),
                    "message": f"{count} recorded incidents ({count / total * 100:.1f}% of total) — well above average ({mean_count:.0f}).",
                    "recommendation": f"Conduct targeted safety review for {category.lower()} hazards. Ensure compliance with relevant DGMS regulations."
                })

        state_deaths = df.groupby('state')['deaths_count'].sum().sort_values(ascending=False)
        mean_deaths = float(state_deaths.mean())
        for state_name, deaths in state_deaths.items():
            if deaths > mean_deaths * 1.8:
                severity = "HIGH" if deaths > mean_deaths * 3 else "MEDIUM"
                alerts.append({
                    "type": "geographic_hotspot",
                    "severity": severity,
                    "title": f"Fatality hotspot: {state_name}",
                    "state": state_name,
                    "deaths": int(deaths),
                    "message": f"{state_name} accounts for {deaths} fatalities — significantly above the per-state average ({mean_deaths:.0f}).",
                    "recommendation": f"Prioritize mine inspections in {state_name}. Review recent enforcement actions."
                })

        mine_deaths = df.groupby('mine_type')['deaths_count'].sum().sort_values(ascending=False)
        if len(mine_deaths) > 0 and mine_deaths.iloc[0] > 0:
            top_type = mine_deaths.index[0]
            top_deaths = int(mine_deaths.iloc[0])
            alerts.append({
                "type": "mine_type_risk",
                "severity": "MEDIUM",
                "title": f"High-risk mine type: {top_type}",
                "mine_type": top_type,
                "deaths": top_deaths,
                "message": f"{top_type.capitalize()} mines account for the highest fatality count ({top_deaths} deaths).",
                "recommendation": f"Review and reinforce safety protocols specific to {top_type} mine operations."
            })

        severity_order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        return sorted(alerts, key=lambda x: severity_order.get(x["severity"], 0), reverse=True)


app = FastAPI(
    title="Mining Safety AI Backend",
    description="API for analyzing mining accidents and providing AI-powered insights."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global db, rag, classifier, reporter, monitor, llm, agent_executor, chat_history
    
    print("Loading database...")
    db = MiningAccidentDatabase(csv_path=CSV_PATH, codes_json_path=CODES_JSON_PATH)
    
    print("Loading RAG system...")
    rag = MiningRAGSystem(database=db, index_path=VECTOR_INDEX_PATH)
    
    print("Training classifier...")
    classifier = AccidentClassifier(database=db)
    classifier.train()
    
    reporter = ReportGenerator(database=db)
    monitor = TrendMonitor(database=db)

    chat_history = []
    
    print("Initializing Google Gemini LLM...")
    llm = None
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("FATAL ERROR: GOOGLE_API_KEY environment variable not set.")
        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key
            )
            llm.invoke("Hello")
            print("Gemini LLM connection successful.")
    
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to Google Gemini. {e}")

    if llm and rag.get_retriever() and classifier.pipeline:
        print("Creating Agent Tools...")

        class ClassifyInput(LangchainBaseModel):
            description: str = Field(description="The full text description of the new accident to classify.")
            
        class ReportInput(LangchainBaseModel):
            year: Optional[int] = Field(default=None, description="The specific year to filter the report for.")
            state: Optional[str] = Field(default=None, description="The specific state to filter the report for.")

        tools = [
            create_retriever_tool(
                rag.get_retriever(),
                "search_accident_reports",
                "Use this tool to answer specific questions about *past* mining accidents. Input should be a query like 'what happened in the Godavari accident?' or 'show me incidents related to methane'."
            ),
            StructuredTool.from_function(
                name="classify_new_accident",
                description="Use this tool to classify a *new* accident description and find its official accident code. The input must be the full description of the accident.",
                func=classifier.predict,
                args_schema=ClassifyInput
            ),
            StructuredTool.from_function(
                name="get_safety_statistics",
                description="Use this tool to get high-level dashboard statistics, like total deaths, number of accidents, or breakdowns by state and category. This tool takes no input.",
                func=db.get_statistics
            ),
            StructuredTool.from_function(
                name="generate_safety_audit_report",
                description="Use this tool to generate a full, text-based safety audit report. You can optionally filter by year or state.",
                func=reporter.create_safety_report,
                args_schema=ReportInput
            )
        ]
        
        system_prompt = """
You are an expert "Digital Mine Safety Officer" operating under the Directorate General of Mines Safety (DGMS), India. Your duty is to provide helpful, accurate, and actionable answers.

You have access to a database of Indian mining accidents and four tools:
1. `search_accident_reports` — Answer questions about *specific past accidents* (e.g., "what happened at Godavari?", "show methane incidents").
2. `get_safety_statistics` — Provide *overall statistics*: total deaths, breakdowns by state/category/mine type.
3. `classify_new_accident` — Classify a *new accident description* the user provides and return its DGMS accident code.
4. `generate_safety_audit_report` — Generate a full audit report, optionally filtered by year or state.

Decision process:
- If the user asks about specific past accidents or patterns → use `search_accident_reports`.
- If the user asks for totals, counts, or breakdowns → use `get_safety_statistics`.
- If the user describes a new incident and wants it classified → use `classify_new_accident`.
- If the user wants a full report → use `generate_safety_audit_report`.
- If the user asks a general mining safety question (regulations, best practices) → answer from your expert knowledge without calling tools.

When recommending regulatory actions, always cite the specific DGMS regulation number from the accident record (e.g., "Regulation 112(2)(C) of the Metalliferous Mines Regulations, 1961"). If regulations are noted in the retrieved record, quote them directly.

When a tool finds no matching records, state that clearly rather than guessing.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True
        )
        print("AI Agent is online.")
        
    else:
        agent_executor = None
        print("Warning: AI Agent is OFFLINE. LLM, Retriever, or Classifier failed to initialize.")

class ChatQuery(BaseModel):
    query: str

class ClassificationQuery(BaseModel):
    description: str

@app.get("/")
def read_root():
    return {"status": "Mining Safety AI Backend is running."}

@app.get("/statistics")
def get_dashboard_statistics():
    """Get high-level statistics for a dashboard"""
    return db.get_statistics()

@app.get("/generate_report", response_class=StreamingResponse)
async def get_safety_report(year: Optional[int] = None, state: Optional[str] = None):
    """Generate a plain-text safety audit report"""
    report = reporter.create_safety_report(year=year, state=state)
    return StreamingResponse(iter([report]), media_type="text/plain")

@app.get("/anomalies")
def get_anomaly_alerts():
    """Get detected anomalies and rising trend alerts from accident data"""
    return monitor.detect_anomalies()


@app.post("/classify_new_accident")
def classify_accident(query: ClassificationQuery):
    """
    (Solves organizer's request)
    Predicts the accident code for a *new* description.
    """
    if not classifier or not classifier.pipeline:
        raise HTTPException(status_code=503, detail="Classifier is not available.")
    
    return classifier.predict(query.description)

@app.get("/updates")
def get_recent_updates():
    """
    Fetches the list of recent updates from the 'updates.json' file.
    """
    try:
        with open(UPDATES_JSON_PATH, 'r') as f:
            updates_data = json.load(f)
        return sorted(updates_data, key=lambda x: x.get('date', '1970-01-01'), reverse=True)
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{UPDATES_JSON_PATH} file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Error decoding {UPDATES_JSON_PATH}.")

@app.post("/chat_stream")
async def chat_with_agent(query: ChatQuery):
    """
    (The new Agent Chatbot)
    Conversational agent endpoint with streaming and memory.
    """
    global agent_executor, chat_history
    
    if not agent_executor:
        raise HTTPException(status_code=503, detail="Chat agent is not available. Check server logs.")
    
    print(f"Streaming agent query: {query.query}")
    
    async def stream_generator():
        global chat_history
        full_response_content = ""
        try:
            input_data = {
                "input": query.query,
                "chat_history": chat_history
            }

            async for event in agent_executor.astream_events(input_data, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content if hasattr(chunk, "content") else ""
                    # Skip tool-call deltas — they have empty string content
                    if isinstance(content, str) and content:
                        yield content
                        full_response_content += content

        except Exception as e:
            print(f"Error during agent stream: {e}")
            yield f"Error: An error occurred: {e}"
        finally:
            if full_response_content:
                chat_history.append(HumanMessage(content=query.query))
                chat_history.append(AIMessage(content=full_response_content))
            chat_history = chat_history[-6:]

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    print("Starting FastAPI server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
