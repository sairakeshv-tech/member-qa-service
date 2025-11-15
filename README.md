# Member Q&A Service

A simple question-answering API that answers questions about member data from a public API. Ask things like "When is Layla planning her trip to London?" and get answers based on the member messages.

## Quick Start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python app.py
```

Then open http://localhost:8000 in your browser. You'll see a simple UI where you can ask questions. The API is also available at `/ask` and there's interactive docs at `/docs`.

### Using the API

You can call it with GET or POST:

```bash
# GET request
curl "http://localhost:8000/ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London?"

# POST request
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars does Vikram Desai have?"}'
```

Response format:
```json
{
  "answer": "Vikram Desai has 2 car(s)."
}
```

## How It Works

The system fetches messages from the API, indexes them, and then when you ask a question:

1. It tries to figure out which user you're asking about (using fuzzy matching on names)
2. Retrieves relevant messages using BM25 ranking + keyword matching
3. Generates an answer based on the question type (temporal, quantitative, or entity extraction)

The code is pretty straightforward - check out `app.py` if you want to see the implementation.

## Dataset

The data comes from a public API with about 3,349 messages from 10 different members. Each message has:
- User name
- Timestamp
- The actual message text

The messages cover things like travel plans, restaurant preferences, car ownership, etc. Pretty clean dataset overall - I didn't find any major issues when I analyzed it.

## Design Notes: Alternative Approaches I Considered

When building this, I thought about a few different approaches. Here's what I considered and why I went with the current one:

### 1. Using an LLM (GPT/Claude) directly

This was tempting - just send all the messages to an LLM and ask the question. It would handle complex questions really well and give natural answers.

**Why I didn't do it**: Cost and latency. Every query would need an API call, and with 3,349 messages, that's a lot of tokens. Also, LLMs can hallucinate, which isn't great for a system that needs to be accurate. Maybe as a future enhancement where I use the LLM just for answer generation after retrieving relevant messages.

### 2. Vector embeddings + semantic search

I could use sentence-transformers to create embeddings and do semantic search. This would be better at understanding meaning, not just keywords.

**Why I didn't do it**: It's slower than BM25 and uses more memory (~500MB for the model). For this use case, keyword matching works pretty well since the questions are fairly specific. I actually have `sentence-transformers` in the requirements but ended up not using it - might add it later if I need better semantic understanding.

### 3. Building a knowledge graph with NER

Extract all entities (people, places, dates, etc.) and build a structured knowledge graph. Then query that graph for answers.

**Why I didn't do it**: Way too complex for this. Would need accurate NER, handle relationships, deal with implicit information. Cool idea but overkill for now. Maybe if I need to answer complex multi-hop questions later.

### 4. Just regex patterns

Keep it super simple - just pattern matching with regex.

**Why I didn't do it**: Too limited. Would miss a lot of relevant messages and struggle with variations in how people phrase things. I do use patterns for answer generation, but combining it with BM25 for retrieval works much better.

### 5. Hybrid: Retrieval + LLM (what I'd do next)

Use the current retrieval system to find relevant messages, then send just those to an LLM to generate a natural answer.

**Why this is interesting**: Best of both worlds. Accurate retrieval (no hallucination from irrelevant context) + natural answer generation. The current implementation is actually a good foundation for this - I could add an LLM layer on top without changing much.

### 6. Current approach: BM25 + pattern matching

What I actually built. BM25 for finding relevant messages, then pattern-based extraction for generating answers.

**Why I chose this**: 
- Fast (no external API calls)
- Works offline
- Predictable behavior
- Good enough for the example questions
- Simple to understand and maintain

The main downside is that it's limited to predefined patterns, so it won't handle every possible question. But for the assignment requirements, it works well.

## Data Insights

I analyzed the dataset and here's what I found:

**Basic stats:**
- 3,349 total messages
- 10 unique users
- Messages are pretty evenly distributed (each user has 288-365 messages)
- Time range: 2024-2025

**Content breakdown:**
- About 266 messages mention trips/travel (8%)
- 180 messages about cars (5%)
- 271 messages about restaurants (8%)

**Data quality:**
The data looks clean. All messages have the required fields, no duplicates, timestamps are valid, and user IDs are consistent. I didn't find any major anomalies.

**Things to watch out for:**
- Users might be referred to by first name, last name, or full name in questions
- Some dates are relative ("this Friday") which need context to interpret
- Some information is implicit (like counting cars by brand mentions)
- Some questions might have multiple valid answers

**What I'd improve:**
- Normalize user names during indexing to handle name variations better
- Add caching so we don't re-fetch all messages on every restart
- Track which question types work well and which don't
- Better handling of edge cases

## Technical Stuff

Built with FastAPI, using BM25 for ranking and some basic NLP for answer generation. The whole thing is pretty lightweight - initializes in 5-10 seconds and queries are usually under 100ms.

**Dependencies:**
- FastAPI for the web framework
- rank-bm25 for retrieval ranking
- httpx for API calls
- numpy for the math
- pydantic for validation

**Limitations:**
- Only handles predefined question patterns
- Doesn't learn from feedback
- Single-turn only (no conversation context)
- No confidence scores

## Deployment

You can deploy this anywhere that runs Python. I've tested it locally, but it should work on:
- Railway or Render (easiest, good free tiers)
- Heroku
- Google Cloud Run
- AWS Lambda (with some modifications)

The `/health` endpoint returns the status and some basic stats, which is useful for monitoring.

## Future Improvements

If I were to continue working on this, I'd probably:
1. Add semantic search with sentence-transformers for better retrieval
2. Use an LLM for answer generation (hybrid approach)
3. Better entity extraction (maybe with spaCy)
4. Add confidence scores so users know how reliable answers are
5. Cache messages to disk so restarts are faster
6. Handle multi-hop questions that need multiple facts
7. Support follow-up questions with conversation context

But for now, this works pretty well for the assignment requirements!
