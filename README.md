# Multi-Agent Public Health Decision Support System

An interactive web application where three AI specialist agents debate the applicability of user-selected public health interventions based on real CDC forecast data, then a critical facilitator exposes trade-offs and forces the policymaker to decide.

## How It Works

1. **Select disease and jurisdiction** — Choose between COVID-19 or Influenza, and pick a U.S. state or national level
2. **View live forecast data** — The system fetches real hospital admission data from the CDC NHSN API and runs an ARIMA forecast showing latest admissions, week-over-week change, 4-week trend, and next-week prediction
3. **Set interventions** — Toggle mask mandate and school closure on/off
4. **Start the discussion** — Click "Start Analysis Discussion", then step through messages one at a time using "Continue Discussion":
   - **Agent introductions** — Each agent briefly introduces their role and expertise
   - **Structured analyses** — Each agent provides a structured assessment using the format: Claim, Evidence (with verified source URL), Assumption, Limitation, and Confidence level
   - **Critical facilitator** — Exposes where agents agree/disagree, identifies blind spots, highlights risks of action vs. inaction, provides a numeric trigger for policy change, and frames the decision as a choice for the policymaker
5. **Join the conversation** — Ask questions, challenge recommendations, or request deeper analysis
6. **Summarize sources** — Click "Summarize" next to any citation to get bullet-point highlights (fetches real page content when accessible, falls back to curated summaries for paywalled sources)

## Agents

| Agent | Role | Focus |
|---|---|---|
| Dr. Amara | Epidemiologist | Transmission patterns, infection curves, whether the outbreak is accelerating or declining |
| Dr. Chen | Healthcare Analyst | Hospital capacity, ICU strain, staffing pressure, surge thresholds |
| Prof. Rivera | Economist | Cost-benefit of restrictions, GDP impact, education loss, workforce disruption |
| Facilitator | Critical Synthesizer | Exposes disagreements, identifies blind spots, challenges consensus, provides numeric policy triggers |

## Design Principles

- **Structured reasoning** — Agents use Claim/Evidence/Assumption/Limitation/Confidence format, not vague opinions
- **Verified evidence** — Each agent draws from a curated list of 6 real, URL-verified sources (CDC, WHO, NEJM, Lancet, IMF, etc.)
- **No fake consensus** — Agents are instructed to disagree when their expertise points differently
- **Uncertainty communication** — Every initial assessment includes a confidence level with justification
- **Critical facilitation** — The facilitator challenges rather than summarizes, presenting both risks of following and ignoring the advice
- **Numeric triggers** — Policy recommendations include specific thresholds for reconsideration (e.g., "reconsider if admissions exceed 50 for 2 weeks")
- **One message at a time** — Policymaker controls the pace via "Continue Discussion" button
- **Evidence on demand** — Sources are cited in the initial round only; follow-up responses are direct and source-free

## Architecture

```
Browser (index.html)  <-->  Flask API (app.py)  <-->  Azure OpenAI (GPT-4)
                                  |
                            CDC NHSN API (real-time hospital admission data)
                                  |
                            ARIMA Forecast (statsmodels)
```

All code is in two files:
- `app.py` — Flask backend with agent orchestration, CDC data fetching, ARIMA forecasting, and all API endpoints
- `frontend/index.html` — Single-page UI with chat interface, intervention controls, and forecast display

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the frontend |
| `/api/jurisdictions` | GET | List all available U.S. jurisdictions |
| `/api/forecast` | GET | Fetch CDC data and return ARIMA forecast (`?disease=covid&jurisdiction=USA`) |
| `/api/start` | POST | Start a new discussion with agent intros, analyses, and facilitator synthesis |
| `/api/chat` | POST | Send a user message, get 2 agent responses |
| `/api/continue` | POST | Let agents continue discussing autonomously |
| `/api/facilitate` | POST | Call the facilitator to synthesize and challenge |
| `/api/summarize` | POST | Summarize a research paper (live fetch or curated fallback) |

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set environment variables

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
```

### Run the application

```bash
python app.py
```

Then open **http://localhost:5001** in your browser.

## Project Structure

```
CS_568/
├── app.py                 # Flask backend (agents, forecasting, API routes)
├── frontend/
│   └── index.html         # Frontend UI (chat, controls, forecast display)
├── requirements.txt       # Python dependencies
└── README.md
```

## Dependencies

- Flask + Flask-CORS (web server)
- OpenAI SDK (Azure OpenAI GPT-4)
- httpx + BeautifulSoup4 (CDC API calls + paper summarization)
- NumPy + statsmodels + SciPy (ARIMA forecasting)
