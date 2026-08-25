"""Streamlit frontend for the Intelligent Document Q&A Assistant."""

import os
import requests
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001/api/v1")

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, citations}]
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


# ── Helpers ────────────────────────────────────────────────────────────────────
def upload_document(file) -> dict:
    response = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), file.type)},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def query_documents(question: str, session_id: str, top_k: int) -> dict:
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "session_id": session_id, "top_k": top_k},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def list_documents() -> list:
    response = requests.get(f"{API_BASE}/documents", timeout=10)
    response.raise_for_status()
    return response.json()


def delete_document(doc_id: str) -> dict:
    response = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=60)
    response.raise_for_status()
    return response.json()


def health_check() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document Q&A")
    st.markdown("---")

    # API status
    if health_check():
        st.success("✅ API connected")
    else:
        st.error("❌ API unreachable — is the backend running?")

    st.markdown("---")

    # Session ID
    st.subheader("🔖 Session")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Identifies your conversation. Change it to start a fresh chat.",
    )
    st.session_state.session_id = session_id

    st.markdown("---")

    # Upload
    st.subheader("📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"],
        help="Upload PDF, DOCX, or TXT files to query against.",
    )

    if uploaded_file:
        if st.button("Upload & Index", type="primary", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    result = upload_document(uploaded_file)
                    doc = result["document"]
                    st.success(
                        f"✅ **{doc['original_filename']}**\n\n"
                        f"{doc['total_chunks']} chunks · {doc.get('total_pages', '?')} pages"
                    )
                    st.session_state.uploaded_docs.append(doc["original_filename"])
                except requests.HTTPError as e:
                    st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")

    # Retrieval settings
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1, max_value=10, value=5,
        help="Number of document chunks retrieved per question.",
    )

    st.markdown("---")

    # Indexed documents
    st.subheader("📚 Indexed Documents")
    try:
        docs = list_documents()
        if docs:
            for doc in docs:
                status_icon = "✅" if doc["status"] == "ready" else "⏳"
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"{status_icon} {doc['original_filename']}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc['original_filename']}"):
                        with st.spinner(f"Deleting {doc['original_filename']}…"):
                            try:
                                delete_document(doc["id"])
                                st.success("Deleted!")
                                st.rerun()
                            except requests.HTTPError as e:
                                st.error(f"Failed: {e.response.json().get('detail', str(e))}")
                            except Exception as e:
                                st.error(f"Failed: {e}")
        else:
            st.caption("No documents uploaded yet.")
    except Exception:
        st.caption("Could not load documents.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("💬 Intelligent Document Q&A Assistant")
st.caption(
    "Upload documents in the sidebar, then ask questions below. "
    "Answers are grounded in your documents with source citations."
)

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📎 Sources", expanded=False):
                for c in msg["citations"]:
                    st.markdown(
                        f"- **{c['document_name']}** — Page {c['page_number']} "
                        f"*(relevance: {c['relevance_score']:.2f})*"
                    )
        if msg.get("latency"):
            st.caption(
                f"⏱ Retrieval: {msg['latency']['retrieval']:.2f}s · "
                f"LLM: {msg['latency']['response']:.2f}s · "
                f"Model: {msg['latency']['model']}"
            )

# Chat input
if question := st.chat_input("Ask a question about your documents…"):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = query_documents(
                    question=question,
                    session_id=st.session_state.session_id,
                    top_k=top_k,
                )
                answer = result["answer"]
                citations = result.get("citations", [])

                st.markdown(answer)

                if citations:
                    with st.expander("📎 Sources", expanded=False):
                        for c in citations:
                            st.markdown(
                                f"- **{c['document_name']}** — Page {c['page_number']} "
                                f"*(relevance: {c['relevance_score']:.2f})*"
                            )

                latency = {
                    "retrieval": result["retrieval_latency_seconds"],
                    "response": result["response_latency_seconds"],
                    "model": result["model_used"],
                }
                st.caption(
                    f"⏱ Retrieval: {latency['retrieval']:.2f}s · "
                    f"LLM: {latency['response']:.2f}s · "
                    f"Model: {latency['model']}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "latency": latency,
                })

            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                error_msg = f"❌ Error: {detail}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except Exception as e:
                error_msg = f"❌ Could not reach the API: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
