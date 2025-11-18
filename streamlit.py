import streamlit as st
import os
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.controller_agent import ControllerAgent
from agents.memory_agent import MemoryAgent
import time

# Page configuration
st.set_page_config(
    page_title="Vision-Enhanced Multi-Agent RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
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
    .intent-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .intent-fact { background-color: #d4edda; color: #155724; }
    .intent-analysis { background-color: #d1ecf1; color: #0c5460; }
    .intent-summary { background-color: #fff3cd; color: #856404; }
    .intent-visual { background-color: #f8d7da; color: #721c24; }
    .answer-box {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .memory-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agents():
    """Initialize all agents (cached to avoid reloading on every interaction)"""
    try:
        intent_agent = IntentAgent()
        retrieval_agent = RetrievalAgent()
        vision_agent = VisionAgent()
        reasoning_agent = ReasoningAgent()
        memory_agent = MemoryAgent(max_turns=12)
        controller = ControllerAgent(
            intent_agent, 
            retrieval_agent, 
            vision_agent, 
            reasoning_agent, 
            memory_agent
        )
        return controller, retrieval_agent, vision_agent, memory_agent
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        st.stop()

def get_intent_color(intent):
    """Get CSS class for intent badge"""
    colors = {
        "fact": "intent-fact",
        "analysis": "intent-analysis",
        "summary": "intent-summary",
        "visual": "intent-visual"
    }
    return colors.get(intent, "intent-fact")

def main():
    # Header
    st.markdown('<div class="main-header">🤖 Vision-Enhanced Multi-Agent RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intent-Aware, Multimodal Retrieval-Augmented Generation System</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'data_ingested' not in st.session_state:
        st.session_state.data_ingested = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙ Configuration")
        
        # System Info
        st.subheader("System Components")
        st.write("✅ *Intent Agent* - ML/DL Classification")
        st.write("✅ *Retrieval Agent* - RAG with ChromaDB")
        st.write("✅ *Vision Agent* - BLIP Image Captioning")
        st.write("✅ *Reasoning Agent* - Mistral LLM")
        st.write("✅ *Memory Agent* - Conversation Context")
        st.write("✅ *Controller Agent* - Orchestration")
        
        st.divider()
        
        # Data Status
        st.subheader("📊 Data Status")
        if os.path.exists("data"):
            pdf_count = len([f for f in os.listdir("data") if f.endswith(".pdf")])
            images_dir = os.path.join("data", "images")
            image_count = len([f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg", ".jpeg"))]) if os.path.exists(images_dir) else 0
            st.success(f"📄 PDFs: {pdf_count}")
            st.success(f"🖼 Images: {image_count}")
        else:
            st.warning("Data directory not found")
        
        st.divider()
        
        # Model Info
        st.subheader("🧠 Models")
        st.info("""
        - *Intent*: Custom PyTorch Model
        - *Embeddings*: Sentence Transformers (all-MiniLM-L6-v2)
        - *VLM*: BLIP Image Captioning
        - *LLM*: Mistral-7B (llama-cpp)
        - *Vector DB*: ChromaDB
        """)
        
        st.divider()
        
        # Intent Types
        st.subheader("🎯 Intent Types")
        st.write("*fact*: Direct factual lookup")
        st.write("*analysis*: Comparative reasoning")
        st.write("*summary*: Document summarization")
        st.write("*visual*: Chart/image queries")
        
        st.divider()
        
        # Settings
        st.subheader("⚙ Query Settings")
        top_k = st.slider("Top K Results", 3, 10, 5)
        
        st.divider()
        
        # Memory Management
        st.subheader("💾 Memory")
        if st.button("Clear Conversation History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Show recent memory
        if 'memory_agent' in st.session_state:
            memory_text = st.session_state.memory_agent.get_context_text(6)
            if memory_text:
                with st.expander("View Recent Memory"):
                    st.text(memory_text)
    
    # Initialize agents
    controller, retrieval_agent, vision_agent, memory_agent = initialize_agents()
    st.session_state.memory_agent = memory_agent
    
    # Data ingestion
    if not st.session_state.data_ingested:
        with st.spinner("Ingesting data from 'data/' directory..."):
            try:
                retrieval_agent.ingest_pdfs(data_dir="data")
                retrieval_agent.ingest_images_folder(images_dir=os.path.join("data", "images"))
                st.session_state.data_ingested = True
                st.success("✅ Data ingestion complete!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.warning(f"Data ingestion note: {e}")
                st.session_state.data_ingested = True  # Continue anyway
    
    # Main query interface
    st.markdown("---")
    st.header("💬 Query Interface")
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'What does the emissions chart show?' or 'Summarize the key findings'",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("🔍 Search", type="primary")
    
    # Process query
    if submit_button and query:
        with st.spinner("Processing query through multi-agent pipeline..."):
            try:
                # Handle query through controller
                res = controller.handle_query(query, top_k=top_k)
                
                # Store in conversation history
                st.session_state.conversation_history.append({
                    "query": query,
                    "intent": res.get("intent", "unknown"),
                    "answer": res.get("answer", ""),
                    "retrieved_count": len(res.get("retrieved", [])),
                    "images_count": len(res.get("images", []))
                })
                
                # Display results
                st.markdown("---")
                
                # Intent Classification
                st.header("🎯 Intent Classification")
                intent = res.get("intent", "unknown")
                intent_classes = {
                    "fact": "Direct Factual Lookup",
                    "analysis": "Multi-document Reasoning",
                    "summary": "Summarization",
                    "visual": "Visual Content Query"
                }
                intent_desc = intent_classes.get(intent, intent)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f'<div class="intent-badge {get_intent_color(intent)}">{intent.upper()}</div>', unsafe_allow_html=True)
                with col2:
                    st.info(f"{intent_desc}: The query has been classified as a *{intent}* type question.")
                
                # Answer
                st.markdown("---")
                st.header("💡 Answer")
                answer = res.get("answer", "No answer generated.")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Retrieved Documents
                st.markdown("---")
                st.header("📚 Retrieved Documents")
                retrieved = res.get("retrieved", [])
                
                if retrieved:
                    for i, r in enumerate(retrieved):
                        meta = r.get("metadata", {})
                        doc_type = meta.get("type", "unknown")
                        source = meta.get("source", "Unknown")
                        page = meta.get("page", "N/A")
                        document = r.get("document", "")
                        distance = r.get("distance", 0)
                        
                        with st.expander(f"📄 Result {i+1}: {source} (Page {page}) - {doc_type.upper()} [Distance: {distance:.4f}]", expanded=(i==0)):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.write(f"*Type*: {doc_type}")
                                st.write(f"*Source*: {source}")
                                st.write(f"*Page*: {page}")
                                st.write(f"*Similarity*: {1-distance:.4f}")
                            with col2:
                                st.write("*Content*:")
                                st.text_area("", document, height=150, key=f"doc_{i}", disabled=True)
                else:
                    st.warning("No documents retrieved.")
                
                # Image Contexts
                images = res.get("images", [])
                if images:
                    st.markdown("---")
                    st.header("🖼 Image Contexts")
                    
                    for idx, im in enumerate(images):
                        meta = im.get("meta", {})
                        img_name = meta.get("img_name", "Unknown")
                        img_path = meta.get("img_path", "")
                        caption = im.get("caption", "")
                        ocr_text = im.get("ocr_text", "")
                        
                        st.subheader(f"Image {idx + 1}: {img_name}")
                        
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=img_name, use_container_width=True)
                            else:
                                st.info(f"Image path: {img_path}")
                        
                        with col2:
                            if caption:
                                st.write("*Caption:*")
                                st.info(caption)
                            if ocr_text:
                                st.write("*OCR Text:*")
                                st.code(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text, language="text")
                        
                        if idx < len(images) - 1:
                            st.divider()
                
                # Memory Context
                memory_used = res.get("memory_used", "")
                if memory_used:
                    st.markdown("---")
                    st.header("💾 Conversation Memory")
                    with st.expander("View Memory Context Used"):
                        st.markdown(f'<div class="memory-box">{memory_used}</div>', unsafe_allow_html=True)
                
                # Agent Pipeline Visualization
                st.markdown("---")
                st.header("🔄 Agent Pipeline")
                st.info(f"""
                *Query Processing Flow:*
                1. *Intent Agent* → Classified query as {intent}
                2. *Retrieval Agent* → Retrieved {len(retrieved)} document(s)
                3. *Vision Agent* → Processed {len(images)} image(s)
                4. *Reasoning Agent* → Generated answer using Mistral LLM
                5. *Memory Agent* → Maintained conversation context
                6. *Controller Agent* → Orchestrated the entire pipeline
                """)
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)
    
    # Conversation History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📜 Conversation History")
        for idx, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
            with st.expander(f"Query {len(st.session_state.conversation_history) - idx}: {entry['query'][:50]}...", expanded=False):
                st.write(f"*Intent*: {entry['intent']}")
                st.write(f"*Answer*: {entry['answer'][:200]}...")
                st.write(f"*Retrieved*: {entry['retrieved_count']} docs, {entry['images_count']} images")
    
    # Example queries
    if not submit_button or not query:
        st.markdown("---")
        st.subheader("💡 Example Queries")
        
        example_cols = st.columns(4)
        examples = [
            ("What does the emissions chart show?", "visual"),
            ("Summarize the key findings", "summary"),
            ("Compare the results between documents", "analysis"),
            ("What is the main topic?", "fact")
        ]
        
        for idx, (example, intent_type) in enumerate(examples):
            with example_cols[idx]:
                if st.button(example, key=f"example_{idx}"):
                    st.session_state.query_input = example
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>Vision-Enhanced Multi-Agent RAG System</strong> | Built with Streamlit</p>
            <p>Components: Intent Classification | RAG | Vision | LLM Reasoning | Memory</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

