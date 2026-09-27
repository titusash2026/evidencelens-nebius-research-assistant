import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from src.report import build_markdown_report

from src.planner import generate_research_plan
from src.safety import get_research_disclaimer, validate_research_question
from src.verifier import verify_claims

load_dotenv()

st.set_page_config(
    page_title="EvidenceLens",
    page_icon="🔬",
    layout="wide",
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

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

st.title("🔬 EvidenceLens")
st.write(
    "A trusted research agent for IPF, medical imaging, and medical-AI literature."
)
st.caption(get_research_disclaimer())

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
    st.header("Research settings")

    source_mode = st.radio(
        "Source policy",
        ["Trusted medical & scientific sources", "Broad web sources"],
    )

    st.info(
        "Trusted mode prioritises PubMed, NCBI, NIH, WHO, journals, "
        "and radiology organisations."
    )

question = st.text_area(
    "Enter your research question",
    placeholder=(
        "Example: What HRCT findings are most consistently associated "
        "with UIP-pattern IPF?"
    ),
    height=120,
)

if st.button("Run EvidenceLens research agent", type="primary"):
    is_valid, safety_message = validate_research_question(question)

    if not is_valid:
        st.warning(safety_message)
        st.stop()

    try:
        with st.status("EvidenceLens agent is working...", expanded=True) as status:
            st.write("1. Planning focused evidence questions with NVIDIA Nemotron...")

            research_plan = generate_research_plan(
                client=nebius_client,
                model=MODEL_NAME,
                question=question,
            )

            st.write("2. Retrieving evidence with Tavily...")

            source_map = {}

            for plan_question in research_plan:
                search_options = {
                    "query": plan_question,
                    "search_depth": "advanced",
                    "max_results": 2,
                }

                if source_mode == "Trusted medical & scientific sources":
                    search_options["include_domains"] = TRUSTED_DOMAINS

                search_response = tavily_client.search(**search_options)

                for source in search_response.get("results", []):
                    url = source.get("url", "")

                    if url and url not in source_map:
                        source_map[url] = source

            sources = list(source_map.values())[:6]

            if not sources:
                status.update(
                    label="No matching evidence sources found",
                    state="error",
                )
            else:
                source_context = "\n\n".join(
                    [
                        f"[{index}] Title: {source.get('title', 'Untitled')}\n"
                        f"URL: {source.get('url', '')}\n"
                        f"Content: {source.get('content', '')}"
                        for index, source in enumerate(sources, start=1)
                    ]
                )

                st.write(
                    "3. Generating a cited research synthesis with NVIDIA Nemotron..."
                )

                response = nebius_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are EvidenceLens, a careful medical-AI research "
                                "assistant. Answer only from the supplied source material. "
                                "Cite every factual claim using [1], [2], and so on. "
                                "If evidence is insufficient, state that clearly. "
                                "If sources conflict, state the conflict clearly. "
                                "Do not diagnose, treat, prescribe, or provide "
                                "patient-specific medical advice."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Research question:\n{question}\n\n"
                                f"Source material:\n{source_context}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=2000,
                )

                summary = response.choices[0].message.content or (
                    "No research summary was generated."
                )

                st.write("4. Verifying summary claims against the evidence...")

                verification_items = verify_claims(
                    client=nebius_client,
                    model=MODEL_NAME,
                    answer=summary,
                    sources=sources,
                )

                status.update(
                    label="EvidenceLens agent completed",
                    state="complete",
                )

        if not sources:
            st.warning(
                "No matching sources found. Try a more specific question "
                "or choose Broad web sources."
            )
            st.stop()

        st.subheader("Agent research plan")
        for index, plan_question in enumerate(research_plan, start=1):
            st.write(f"{index}. {plan_question}")

        st.subheader("Evidence-based research summary")
        st.write(summary)

        st.subheader("Claim verification")

        if verification_items:
            st.dataframe(
                verification_items,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "The verifier could not produce structured results for this run. "
                "Please review the cited sources directly."
            )
            report_content = build_markdown_report(
            question=question,
            research_plan=research_plan,
            summary=summary,
            verification_items=verification_items,
            sources=sources,
        )

        st.download_button(
            label="Download evidence report",
            data=report_content,
            file_name="evidencelens-evidence-report.md",
            mime="text/markdown",
        )

        st.subheader("Sources used")
        for index, source in enumerate(sources, start=1):
            title = source.get("title", "Untitled source")
            url = source.get("url", "")
            st.markdown(f"{index}. [{title}]({url})")

    except Exception as error:
        st.error(f"Something went wrong: {error}")