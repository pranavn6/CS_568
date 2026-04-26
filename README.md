# Multi-Agent Public Health Decision Support System

An interactive web application where three AI specialist agents discuss the applicability of user-selected public health interventions based on real CDC forecast data.

## How It Works

1. **Select disease and jurisdiction** — Choose between COVID-19 or Influenza, and pick a U.S. state or national level
2. **View live forecast data** — The system fetches real hospital admission data from the CDC NHSN API and runs an ARIMA forecast showing latest admissions, week-over-week change, 4-week trend, and next-week prediction
3. **Set interventions** — Toggle mask mandate and school closure on/off
4. **Start the discussion** — Three specialist agents analyze whether your chosen interventions are appropriate given the forecast:
   - **Dr. Amara (Epidemiologist)** — Assesses interventions from a transmission-control perspective
   - **Dr. Chen (Healthcare Analyst)** — Evaluates whether hospitals can handle the projected admissions
   - **Prof. Rivera (Economist)** — Analyzes the cost-benefit trade-off of the restrictions
5. **Join the conversation** — Ask questions, challenge recommendations, or request deeper analysis. Each agent cites credible sources with links, and you can click "Summarize" next to any citation to get bullet-point highlights
6. **Facilitator** — A fourth agent synthesizes agreement/disagreement and guides the discussion

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
| `/api/forecast?disease=covid&jurisdiction=USA` | GET | Fetch CDC data and return ARIMA forecast |
| `/api/start` | POST | Start a new discussion with initial agent analyses |
| `/api/chat` | POST | Send a user message, get 2 agent responses |
| `/api/continue` | POST | Let agents continue discussing autonomously |
| `/api/facilitate` | POST | Call the facilitator to synthesize |
| `/api/summarize` | POST | Summarize a research paper from its URL |

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
- httpx (CDC API calls)
- NumPy + statsmodels + SciPy (ARIMA forecasting)
