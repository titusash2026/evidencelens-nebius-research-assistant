import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

st.set_page_config(
    page_title="Nebius Research Assistant",
    page_icon="🔬",
    layout="wide",
)

TRUSTED_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "who.int",
    "nature.com",
    "thelancet.com",
    "rsna.org",
    "radiologyinfo.org",
]

st.title("🔬 Nebius Research Assistant")
st.write("Search credible research sources and create a source-grounded AI summary.")
st.caption("For research and education only — not medical diagnosis or clinical advice.")

nebius_key = os.getenv("NEBIUS_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

if not nebius_key or not tavily_key:
    st.error("Missing API key. Check that both required keys exist in .env.")
    st.stop()

nebius_client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=nebius_key,
)

tavily_client = TavilyClient(api_key=tavily_key)

with st.sidebar:
    st.header("Search settings")
    source_mode = st.radio(
        "Source policy",
        ["Trusted medical & scientific sources", "Broad web sources"],
    )
    st.info("Trusted mode prioritises PubMed, NCBI, NIH, WHO, journals, and radiology organisations.")

question = st.text_area(
    "Enter your research question",
    placeholder="Example: Why is patient-level splitting important in medical imaging AI?",
    height=120,
)

if st.button("Search sources and generate answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    try:
        search_options = {
            "query": question,
            "search_depth": "advanced",
            "max_results": 5,
        }

        if source_mode == "Trusted medical & scientific sources":
            search_options["include_domains"] = TRUSTED_DOMAINS

        with st.spinner("Searching sources..."):
            search_response = tavily_client.search(**search_options)

        sources = search_response.get("results", [])

        if not sources:
            st.warning(
                "No matching sources found. Try a more specific question or choose Broad web sources."
            )
            st.stop()

        source_context = "\n\n".join(
            [
                f"[{index}] Title: {source.get('title', 'Untitled')}\n"
                f"URL: {source.get('url', '')}\n"
                f"Content: {source.get('content', '')}"
                for index, source in enumerate(sources, start=1)
            ]
        )

        with st.spinner("Nebius is writing a source-grounded answer..."):
            response = nebius_client.chat.completions.create(
                model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful medical-AI research assistant. "
                            "Answer only from the supplied source material. "
                            "Cite each factual claim using [1], [2], and so on. "
                            "If evidence is insufficient or sources conflict, state that clearly. "
                            "Do not diagnose, treat, or give patient-specific medical advice."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question}\n\n"
                            f"Source material:\n{source_context}"
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=2000,
            )

        st.subheader("Research summary")
        st.write(response.choices[0].message.content)

        st.subheader("Sources used")
        for index, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled source")
            url = source.get("url", "")
            st.markdown(f"{index}. [{title}]({url})")

    except Exception as error:
        st.error(f"Something went wrong: {error}")