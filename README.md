# LLM Hallucination Tester 🔬

I built this over a weekend because I was tired of LLMs confidently stating absolute nonsense as fact. When building RAG pipelines, you want some way to benchmark how often a model just makes stuff up. This tool generates a bunch of factual questions about a topic, gets the model to answer them, and then uses a separate, clean LLM instance to check the answers and score the accuracy.

It also comes with a nice little dark-mode dashboard so you don't have to keep reading raw JSON in the terminal.

---

## How It Works Under the Hood

The app runs a 3-stage pipeline using LangChain:

1. **Question Generation:** The model generates `N` factual, verifiable questions about the topic you gave it.
2. **Answer Generation:** The model answers all those questions (it runs these concurrently so it doesn't take forever).
3. **Fact-Checking:** A separate LLM call evaluates each answer, scoring it as `ACCURATE`, `HALLUCINATED`, or `UNCERTAIN` and gives a short reasoning.

---

## Tech Stack

* **Backend:** Python, FastAPI, LangChain (specifically `langchain-openai`)
* **Frontend:** Vanilla HTML & CSS (served directly by FastAPI)
* **Deployment/Container:** Docker, GCP Cloud Run configuration

---

## Setup & Running Locally

### Prerequisites

* Python 3.11+
* An OpenAI API key

### 1. Local Setup

First, clone this thing and go into the folder:

```bash
git clone https://github.com/aayeraahmad531/llm-hallucination-tester.git
cd llm-hallucination-tester
```

Create a virtual env and activate it:

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and paste your OpenAI key in `OPENAI_API_KEY`.

### 3. Run the Server

Start it up with uvicorn:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Now open **http://localhost:8080** in your browser to use the dashboard!
If you just want the raw API Swagger docs, they are at **http://localhost:8080/docs**.

---

## Running with Docker

If you don't want to deal with Python environments, just build the Docker image:

```bash
# Build it
docker build -t hallucination-tester .

# Run it (makes sure it reads your local .env file)
docker run -p 8080:8080 --env-file .env hallucination-tester
```

---

## API Quick Reference

If you want to query it programmatically instead of using the UI:

### `POST /check-hallucination`

**Request:**
```json
{
  "topic": "Discovery of Radium",
  "num_questions": 3,
  "model": "gpt-4o-mini"
}
```

**Response Summary:**
```json
{
  "topic": "Discovery of Radium",
  "questions_tested": 3,
  "hallucination_rate": 0.0,
  "results": [
    {
      "question": "Who discovered radium?",
      "answer": "Marie and Pierre Curie discovered radium.",
      "verdict": "ACCURATE",
      "confidence": 1.0,
      "reasoning": "Marie and Pierre Curie discovered radium in 1898. This is a well-known historical fact."
    }
  ],
  "summary": "Tested 3 questions about 'Discovery of Radium'. Results: 3 accurate, 0 hallucinated, 0 uncertain. Hallucination rate: 0%."
}
```

---

## Known Limitations & Future Plans

* **Rate Limits:** Right now, the async gathering is pretty aggressive. If you ask for 10 questions, it fires them all off at once. If you run this multiple times in a row, you're going to hit OpenAI rate limit errors. I need to add a semaphore or some queue to throttle requests.
* **Anthropic/Gemini Support:** Currently, it's hardcoded to OpenAI since that's what I needed it for, but I want to add dropdown options for Claude and Gemini to compare them.
* **Cost:** Stage 3 runs fact-checking calls per question, which can add up if you run large question batches on expensive models.
