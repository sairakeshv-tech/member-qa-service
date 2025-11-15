"""
Question-Answering System for Member Data
Simple API service that answers natural-language questions about member messages
"""

import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from difflib import SequenceMatcher
from collections import defaultdict, Counter

import httpx
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel

# Try to import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Configuration
API_URL = "https://november7-730026606190.europe-west1.run.app/messages/"


class DataFetcher:
    """Fetches all messages from API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def fetch_all_messages(self) -> List[Dict]:
        """Fetch all messages from API"""
        messages = []
        skip = 0
        limit = 100
        
        print("📡 Fetching messages from API...")
        
        with httpx.Client(timeout=120.0) as client:
            while True:
                try:
                    response = client.get(self.api_url, params={"skip": skip, "limit": limit})
                    
                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("items", [])
                        
                        if not items:
                            break
                        
                        messages.extend(items)
                        print(f"📥 Fetched {len(messages)} messages...")
                        
                        if len(items) < limit:
                            break
                        
                        skip += limit
                    elif response.status_code == 404:
                        break
                    else:
                        print(f"⚠️  Status {response.status_code}, retrying...")
                        import time
                        time.sleep(2)
                        continue
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
                    break
        
        print(f"✅ Fetched {len(messages)} messages total")
        return messages


class UserMatcher:
    """Matches user names from questions"""
    
    def __init__(self, user_index: Dict[str, List[Dict]]):
        self.user_index = user_index
    
    def match(self, question: str) -> Optional[str]:
        """Find user name in question"""
        question_lower = question.lower()
        
        for user_name in self.user_index.keys():
            name_parts = user_name.split()
            for part in name_parts:
                if len(part) > 3 and part in question_lower:
                    return user_name
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        question_words = question_lower.split()
        
        for user_name in self.user_index.keys():
            for name_part in user_name.split():
                if len(name_part) > 3:
                    for qword in question_words:
                        if len(qword) > 3:
                            similarity = SequenceMatcher(None, name_part, qword).ratio()
                            if similarity > 0.7 and similarity > best_score:
                                best_score = similarity
                                best_match = user_name
        
        return best_match


class MessageRetriever:
    """Retrieves relevant messages using BM25 + keyword matching"""
    
    def __init__(self, messages: List[Dict], user_index: Dict[str, List[Dict]]):
        self.messages = messages
        self.user_index = user_index
        self.user_matcher = UserMatcher(user_index)
        
        # Build BM25 index
        if BM25_AVAILABLE:
            tokenized = []
            for m in messages:
                text = m.get("message", "").lower()
                tokens = [t for t in text.split() if len(t) > 1]
                tokenized.append(tokens)
            self.bm25 = BM25Okapi(tokenized, k1=1.2, b=0.75)
        else:
            self.bm25 = None
    
    def retrieve(self, question: str, top_k: int = 10) -> List[Dict]:
        """Retrieve top K relevant messages"""
        user_name = self.user_matcher.match(question)
        candidates = self.user_index.get(user_name.lower(), []) if user_name else self.messages
        
        if not candidates:
            return []
        
        # Get indices
        candidate_indices = [i for i, m in enumerate(self.messages) if m in candidates]
        
        question_lower = question.lower()
        scored = []
        
        # BM25 scores
        bm25_scores = None
        if self.bm25 and candidate_indices:
            tokens = [t for t in question_lower.split() if len(t) > 1]
            all_scores = self.bm25.get_scores(tokens)
            bm25_scores = np.array([all_scores[i] for i in candidate_indices])
            if bm25_scores.max() > bm25_scores.min():
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # Keyword scores
        question_words = set(question_lower.split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
        keywords = question_words - stop_words
        
        for idx, msg in enumerate(candidates):
            msg_text = msg.get("message", "").lower()
            msg_words = set(msg_text.split())
            
            overlap = keywords & msg_words
            keyword_score = len(overlap) / len(keywords) if keywords else 0
            
            # Phrase bonus
            phrase_bonus = 0
            words = question_lower.split()
            for i in range(len(words) - 1):
                if f"{words[i]} {words[i+1]}" in msg_text:
                    phrase_bonus += 0.3
            
            keyword_total = keyword_score + phrase_bonus
            
            # Combine
            if bm25_scores is not None:
                total = 0.6 * float(bm25_scores[idx]) + 0.4 * keyword_total
            else:
                total = keyword_total
            
            if total > 0:
                scored.append((total, msg))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [msg for _, msg in scored[:top_k]]


class AnswerGenerator:
    """Generates answers from messages"""
    
    def generate(self, question: str, messages: List[Dict], user_name: Optional[str] = None) -> str:
        """Generate answer based on question type"""
        question_lower = question.lower()
        user = user_name.split()[0].title() if user_name else "the member"
        
        if "when" in question_lower:
            return self._temporal_answer(question, messages, user)
        elif "how many" in question_lower:
            return self._quantitative_answer(question, messages, user)
        elif "what" in question_lower or "which" in question_lower:
            return self._entity_answer(question, messages, user)
        else:
            return self._generic_answer(messages, user)
    
    def _temporal_answer(self, question: str, messages: List[Dict], user: str) -> str:
        """Answer temporal questions"""
        question_lower = question.lower()
        locations = ["london", "paris", "tokyo", "milan", "monaco", "santorini", "singapore", "new york"]
        location = next((loc for loc in locations if loc in question_lower), None)
        
        trip_keywords = ["trip", "travel", "fly", "book", "flight", "going", "visit", "planning"]
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})',
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        ]
        
        best_msg = None
        best_score = 0
        
        for msg in messages:
            text = msg.get("message", "").lower()
            score = 0
            
            if location and location in text:
                score += 3
            if any(kw in text for kw in trip_keywords):
                score += 1
            if any(re.search(p, text) for p in date_patterns):
                score += 1
            
            if score > best_score:
                best_score = score
                best_msg = msg
        
        if best_msg:
            timestamp = best_msg.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_str = dt.strftime("%B %d, %Y")
            except:
                date_str = timestamp.split('T')[0]
            
            if location:
                return f"{user} is planning a trip to {location.title()} mentioned in a message from {date_str}."
            return f"{user} has a trip mentioned in a message from {date_str}."
        
        return f"I couldn't find trip information for {user}."
    
    def _quantitative_answer(self, question: str, messages: List[Dict], user: str) -> str:
        """Answer quantitative questions"""
        if "car" in question.lower():
            car_msgs = [m for m in messages if any(t in m.get("message", "").lower() 
                                                   for t in ["car", "vehicle", "tesla", "bmw", "mercedes"])]
            if car_msgs:
                numbers = []
                
                for msg in car_msgs:
                    text = msg.get("message", "").lower()
                    patterns = [
                        r'\b(\d+)\s*(?:car|vehicle)(?:s)?\b',
                        r'(?:have|own)\s+(\d+)\s+car',
                        r'(\d+)\s+(?:tesla|bmw|mercedes|ferrari|porsche)',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        if matches:
                            numbers.extend([int(n) for n in matches])
                    
                    # Count brands
                    brands = ["tesla", "bmw", "mercedes", "ferrari", "porsche"]
                    brand_count = sum(1 for b in brands if b in text)
                    if brand_count > 0 and any(p in text for p in ["and a", "and an"]):
                        numbers.append(brand_count)
                
                if numbers:
                    final = max(set(numbers), key=numbers.count) if len(set(numbers)) > 1 else max(numbers)
                    return f"{user} has {final} car(s)."
                else:
                    return f"{user} mentions having cars, but the exact number isn't specified."
            else:
                return f"I don't have messages from {user} mentioning cars."
        
        return f"I couldn't determine the quantity for {user}."
    
    def _entity_answer(self, question: str, messages: List[Dict], user: str) -> str:
        """Answer entity extraction questions"""
        if "restaurant" in question.lower():
            rest_msgs = [m for m in messages if any(t in m.get("message", "").lower() 
                                                  for t in ["restaurant", "dining", "dinner"])]
            if rest_msgs:
                restaurants = set()
                
                for msg in rest_msgs:
                    text = msg.get("message", "")
                    patterns = [
                        r'\bat\s+([A-Z][a-zA-Z\s&\'-]+?)(?:\s|$|\.|,)',
                        r'reservation\s+at\s+([A-Z][a-zA-Z\s&\'-]+?)',
                        r'dinner\s+at\s+([A-Z][a-zA-Z\s&\'-]+?)',
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, text)
                        for match in matches:
                            name = match.group(1).strip()
                            if len(name) > 3 and name.lower() not in ['the', 'a', 'an']:
                                restaurants.add(name)
                
                if restaurants:
                    rest_list = ', '.join(list(restaurants)[:10])
                    return f"{user} has mentioned these restaurants: {rest_list}."
                else:
                    return f"I found restaurant messages from {user}, but names weren't clearly specified."
            else:
                return f"I don't have messages from {user} mentioning restaurants."
        
        return f"I couldn't find entity information for {user}."
    
    def _generic_answer(self, messages: List[Dict], user: str) -> str:
        """Generic answer"""
        if messages:
            text = messages[0].get('message', '').strip()
            if len(text) > 200:
                text = text[:200] + "..."
            return f"Based on messages from {user}: {text}"
        return f"I couldn't find information for {user}."


class QASystem:
    """Main Q&A system"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.messages: List[Dict] = []
        self.user_index: Dict[str, List[Dict]] = {}
        self.retriever: Optional[MessageRetriever] = None
        self.answer_generator = AnswerGenerator()
        self._initialized = False
    
    def initialize(self):
        """Initialize system"""
        if self._initialized:
            return
        
        print("🚀 Initializing Q&A System...")
        try:
            fetcher = DataFetcher(self.api_url)
            self.messages = fetcher.fetch_all_messages()
            
            if not self.messages:
                print("⚠️  No messages loaded")
                self._initialized = True
                return
            
            # Build user index
            self.user_index = defaultdict(list)
            for msg in self.messages:
                user = msg.get("user_name", "").lower()
                self.user_index[user].append(msg)
            
            # Build retriever
            print(f"📊 Building index for {len(self.messages)} messages...")
            self.retriever = MessageRetriever(self.messages, self.user_index)
            
            self._initialized = True
            print(f"✅ Ready: {len(self.messages)} messages, {len(self.user_index)} users")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = True
    
    def answer(self, question: str) -> str:
        """Answer a question"""
        if not question or not question.strip():
            return "Please provide a question."
        
        if not self._initialized:
            self.initialize()
        
        if not self.messages:
            return "Service is initializing. Please try again."
        
        user_name = self.retriever.user_matcher.match(question)
        messages = self.retriever.retrieve(question, top_k=10)
        
        if not messages:
            return "I couldn't find relevant information to answer your question."
        
        return self.answer_generator.generate(question, messages, user_name)


# FastAPI app
qa_system = QASystem(API_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    print("🚀 Starting up...")
    qa_system.initialize()
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Member Q&A Service",
    description="Question-Answering System",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    """Serve UI"""
    return FileResponse('static/index.html')

@app.get("/ask", response_model=AnswerResponse)
async def ask_question(question: str = Query(..., description="The question to answer")):
    """Answer a question"""
    try:
        if not qa_system._initialized:
            return AnswerResponse(answer="Service is still initializing. Please wait a moment and try again.")
        answer = qa_system.answer(question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question_post(request: QuestionRequest):
    """Answer a question (POST)"""
    try:
        answer = qa_system.answer(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy" if qa_system._initialized else "initializing",
        "total_messages": len(qa_system.messages),
        "unique_users": len(qa_system.user_index),
        "initialized": qa_system._initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
