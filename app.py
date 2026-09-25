import os
import json
import time
import streamlit as st
from duckduckgo_search import DDGS
from google import genai

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="NexusAI | Research Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .header-box {
        padding: 1.5rem;
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_report" not in st.session_state:
    st.session_state.current_report = ""
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# ==========================================
# 2. SIDEBAR SETTINGS
# ==========================================
with st.sidebar:
    st.title("⚙️ Engine Settings")
    
    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Enter Gemini API Key:", value=env_key, type="password", key="gemini_key_input")
    
    max_sources = st.slider("Max Search Sources", min_value=2, max_value=10, value=5, key="max_sources_slider")
    
    MODEL_NAME = st.selectbox(
        "LLM Backbone",
        options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.6-flash"],
        index=0
    )
    
    st.divider()
    st.markdown("**📂 File Upload (Optional)**")
    uploaded_files = st.file_uploader(
        "Attach documents or images:",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

if not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def call_gemini_with_retry(prompt: str, model: str, max_retries: int = 4, delay: int = 2) -> str:
    """Handles temporary server load (503) errors with automatic retries."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                st.warning(f"⏳ Server demand high (503). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def fetch_web_sources(query: str, max_results: int = 5):
    """Fetches search snippets safely using DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            for res in search_results:
                results.append({
                    "title": res.get("title", ""),
                    "href": res.get("href", ""),
                    "body": res.get("body", "")
                })
    except Exception:
        pass
    return results

def run_single_gemini_request(query: str, model: str, sources: list, uploaded_files=None):
    """Executes spellcheck, report generation, and follow-ups in ONE single API call with links."""
    if sources:
        context_str = "\n\n".join([
            f"Source Title: {s['title']}\nURL: {s['href']}\nContent Snippet: {s['body']}"
            for s in sources
        ])
    else:
        context_str = "No external web sources available. Generate a thorough briefing using internal knowledge base."

    file_notes = f"\nUser attached {len(uploaded_files)} extra files/images." if uploaded_files else ""

    prompt = f"""
    You are an AI Research Assistant. Analyze the user's query and context below.

    USER QUERY: {query}
    {file_notes}

    CONTEXT / SOURCES:
    {context_str}

    INSTRUCTIONS FOR LINKS:
    - Embed Markdown hyperlinked citations (e.g. [Source Title](URL)) directly in the report text wherever relevant facts are cited.

    Return your response strictly in raw JSON format matching this exact schema:
    {{
      "corrected_query": "Spell-corrected version of the query (or original if correct)",
      "report": "Formal Executive Briefing with Overview, Key Findings (bulleted), and Practical Implications in Markdown format, incorporating inline links where relevant.",
      "followup_questions": ["Question 1", "Question 2", "Question 3"]
    }}
    Do NOT enclose the output in markdown codeblocks (no ```json). Output raw JSON string directly.
    """

    raw_response = call_gemini_with_retry(prompt, model=model)
    clean_text = raw_response.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_text)

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.markdown("""
    <div class="header-box">
        <h1 style='color: #f8fafc; margin:0;'>🔍 NexusAI Research Generator</h1>
        <p style='color: #94a3b8; margin:5px 0 0 0;'>Retrieval-Augmented Generation Dashboard powered by Web Search & Gemini</p>
    </div>
""", unsafe_allow_html=True)

query = st.text_input("Enter your research topic or question:", placeholder="e.g., What is biology")

if st.button("🚀 Run Research Agent", type="primary"):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner(f"Searching web sources for '{query}'..."):
            sources = fetch_web_sources(query, max_results=max_sources)
            st.session_state.current_sources = sources
        
        if not sources:
            st.info(f"ℹ️ Web search yielded no direct results. Generating answer using `{MODEL_NAME}` internal knowledge...")

        with st.spinner(f"Generating research report using {MODEL_NAME}..."):
            try:
                data = run_single_gemini_request(query, MODEL_NAME, sources, uploaded_files)
                
                corrected = data.get("corrected_query", query)
                if corrected.lower() != query.lower():
                    st.info(f"✏️ Auto-corrected search term to: **\"{corrected}\"**")

                st.session_state.current_report = data.get("report", "")
                st.session_state.suggested_questions = data.get("followup_questions", [])
            except Exception as err:
                if "503" in str(err) or "UNAVAILABLE" in str(err):
                    st.error(f"⚠️ `{MODEL_NAME}` is experiencing heavy traffic on Google's end. Switch to `gemini-2.5-flash` in the sidebar and try again!")
                elif "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err):
                    st.error("⚠️ Daily quota limit reached! Please wait or generate a new API key in Google AI Studio.")
                else:
                    st.error(f"Failed to process request: {err}")

# ==========================================
# 5. DISPLAY REPORT RESULTS
# ==========================================
if st.session_state.current_report:
    st.divider()
    st.subheader("📑 Executive Research Report")
    st.markdown(st.session_state.current_report)
    
    # Web Sources Section
    if st.session_state.current_sources:
        st.write("---")
        st.markdown("### 🔗 Referenced Web Sources")
        for src in st.session_state.current_sources:
            title = src.get("title") or "Web Source"
            href = src.get("href", "")
            if href:
                st.markdown(f"- [{title}]({href})")

    # Suggested Follow-up Directions
    if st.session_state.suggested_questions:
        st.write("---")
        st.subheader("💡 Suggested Next Steps / Exploration")
        for q_text in st.session_state.suggested_questions:
            st.markdown(f"- {q_text}")

    # Export & Copy Tools
    st.write("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="📥 Download Report (.md)",
            data=st.session_state.current_report,
            file_name="research_report.md",
            mime="text/markdown"
        )
    with col2:
        with st.expander("📋 Click here to copy raw text"):
            st.code(st.session_state.current_report, language="markdown")