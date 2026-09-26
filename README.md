# EvidenceLens: Trusted Medical Research Assistant

EvidenceLens is a source-grounded research assistant for medical-AI and imaging research. It searches credible public sources, then uses an NVIDIA model through Nebius Token Factory to generate a cited research summary.

> For research and education only. It does not provide medical diagnosis, treatment, or patient-specific advice.

## What it does

1. Accepts a research question.
2. Searches public web sources with Tavily.
3. Offers a trusted-source mode prioritising PubMed, PMC, NCBI, NIH, WHO, journals, and radiology organisations.
4. Sends the retrieved source material to NVIDIA Nemotron through Nebius Token Factory.
5. Produces a cited summary and clickable source list.

## Technology

- **Frontend:** Streamlit
- **Web retrieval:** Tavily Search API
- **AI inference:** Nebius Token Factory
- **NVIDIA model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`

## How it works

```text
Research question
      ↓
Tavily source retrieval
      ↓
Trusted-source filtering
      ↓
Nebius Token Factory + NVIDIA Nemotron
      ↓
Cited research summary + source links
```

## Local setup

```bash
git clone <your-repository-url>
cd nebius-research-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
NEBIUS_API_KEY=your_nebius_key
TAVILY_API_KEY=your_tavily_key
```

Run the app:

```bash
streamlit run app.py
```

## Safety and limitations

- Results depend on the retrieved public sources.
- Citations should be checked before using the information in academic work.
- The tool is not a substitute for clinical judgement or professional medical advice.

## License

MIT License