
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time

# Page config
st.set_page_config(
    page_title="Materials Tech RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
@st.cache_resource
def load_env():
    load_dotenv()
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'qdrant_url': os.getenv('QDRANT_URL'),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY')
    }

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    env_vars = load_env()
    client = QdrantClient(
        url=env_vars['qdrant_url'],
        api_key=env_vars['qdrant_api_key'],
        timeout=60
    )
    return client, env_vars

# Initialize models
@st.cache_resource
def init_models():
    env_vars = load_env()
    embeddings_model = OpenAIEmbeddings(
        model='text-embedding-3-small',
        openai_api_key=env_vars['openai_api_key']
    )
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.1,
        openai_api_key=env_vars['openai_api_key'],
        max_tokens=2500  # Increased for longer, more detailed answers
    )
    return embeddings_model, llm

# Query function
def ask_question(question, client, embeddings_model, llm, collection_name="materials_tech_docs", top_k=3):
    """Query the RAG system"""
    try:
        # Get embedding
        with st.spinner("🔍 Searching knowledge base..."):
            query_vector = embeddings_model.embed_query(question)
        
        # Search Qdrant
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        if not results:
            return "No relevant documents found.", []
        
        # Build context
        context_texts = [r.payload['text'] for r in results]
        sources = []
        for r in results:
            source = r.payload.get('source', 'Unknown')
            page = r.payload.get('page', 0)
            sources.append(f"{source.split('/')[-1]} (Page {page})")
        
        context = "\n\n".join(context_texts)
        
        # Generate answer
        with st.spinner("💭 Generating answer..."):
            prompt = f"""You are an expert Materials Science and Engineering consultant with deep knowledge of ISO/DIN standards, metallography, and materials testing.

Based ONLY on the context provided below, answer the question with 100% confidence and accuracy. 

Your answer must be:
- DETAILED and COMPREHENSIVE (aim for 150-200 words minimum)
- PRECISE with specific technical details, numbers, standards, and procedures
- STRUCTURED with clear paragraphs
- Include relevant ISO/DIN standard numbers and specifications
- Explain WHY and HOW, not just WHAT
- Use technical terminology appropriately

If the context doesn't contain enough information to answer fully, state what IS known based on the context.

Context:
{context}

Question: {question}

Detailed Answer:"""
            
            answer = llm.invoke(prompt).content
        
        return answer, sources
    
    except Exception as e:
        return f"Error: {str(e)}", []

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Materials Technology RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about ISO/DIN standards, metallography, hardness testing, and material properties</p>', unsafe_allow_html=True)
    
    # Initialize
    try:
        client, env_vars = init_qdrant()
        embeddings_model, llm = init_models()
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        st.info("Please check your .env file has OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        collection_name = st.text_input("Collection Name", value="materials_tech_docs")
        top_k = st.slider("Number of sources to retrieve", 1, 10, 5)  # Increased default to 5 for more context
        
        st.markdown("---")
        
        st.header("📊 System Info")
        try:
            collection_info = client.get_collection(collection_name)
            st.success(f"✅ Connected to Qdrant")
            st.metric("Total Documents", f"{collection_info.points_count:,}")
            st.metric("Vector Dimension", collection_info.config.params.vectors.size)
        except Exception as e:
            st.error(f"❌ Collection not found")
        
        st.markdown("---")
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are differences between Brinell and Vickers hardness testing?",
            "How is grain size measured according to ISO 643?",
            "What are mechanical properties of EN-AC44300 aluminum alloy?",
            "What is austenitic grain size?",
            "Explain SEM imaging techniques"
        ]
        
        for i, q in enumerate(sample_questions, 1):
            if st.button(f"{i}. {q[:40]}...", key=f"sample_{i}"):
                st.session_state.question = q
        
        st.markdown("---")
        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        - **LLM**: GPT-3.5-Turbo
        - **Embeddings**: text-embedding-3-small
        - **Vector DB**: Qdrant Cloud
        - **Framework**: LangChain
        """)
    
    # Main area
    st.markdown("---")
    
    # Question input
    question = st.text_input(
        "🔍 Ask your question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g., What is the Vickers hardness test procedure according to ISO standards?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Process query
    if search_button and question:
        st.session_state.question = question
        
        start_time = time.time()
        answer, sources = ask_question(
            question,
            client,
            embeddings_model,
            llm,
            collection_name,
            top_k
        )
        elapsed_time = time.time() - start_time
        
        # Display answer
        st.markdown("### 💡 Answer")
        with st.container():
            st.markdown("""
            <style>
            .answer-container {
                background-color: #f0f8ff;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
                margin: 1rem 0;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown(answer)  # Display answer as plain markdown
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display sources
        if sources:
            st.markdown("### 📚 Sources")
            st.markdown('<div class="source-box">', unsafe_allow_html=True)
            for i, source in enumerate(sources, 1):
                st.markdown(f"**{i}.** {source}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Response Time", f"{elapsed_time:.2f}s")
        with col2:
            st.metric("📄 Sources Used", len(sources))
        with col3:
            st.metric("💬 Answer Length", f"{len(answer.split())} words")
    
    elif search_button:
        st.warning("⚠️ Please enter a question!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔬 Materials Technology RAG System | Powered by OpenAI & Qdrant</p>
        <p>💡 Ask questions about ISO/DIN standards, metallography, and material testing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
