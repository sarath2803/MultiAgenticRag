import streamlit as st
import os
from dotenv import load_dotenv
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.controller_agent import ControllerAgent
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent RAG System",
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agents():
    """Initialize all agents (cached to avoid reloading on every interaction)"""
    with st.spinner("Initializing agents..."):
        try:
            intent_agent = IntentAgent()
            retrieval_agent = RetrievalAgent()
            vision_agent = VisionAgent()
            reasoning_agent = ReasoningAgent()
            controller = ControllerAgent(intent_agent, retrieval_agent, vision_agent, reasoning_agent)
            return controller, retrieval_agent, vision_agent
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
    st.markdown('<div class="main-header">🤖 Multi-Agent RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intent-Driven, Multimodal Retrieval-Augmented Generation</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System Info
        st.subheader("System Components")
        st.write("✅ Intent Agent (ML/DL)")
        st.write("✅ Retrieval Agent (RAG)")
        st.write("✅ Vision Agent (VLM)")
        st.write("✅ Reasoning Agent (LLM)")
        st.write("✅ Controller Agent")
        
        st.divider()
        
        # Data Status
        st.subheader("📊 Data Status")
        if os.path.exists("data"):
            pdf_count = len([f for f in os.listdir("data") if f.endswith(".pdf")])
            image_count = len([f for f in os.listdir(os.path.join("data", "images")) if f.endswith((".png", ".jpg", ".jpeg"))]) if os.path.exists(os.path.join("data", "images")) else 0
            st.success(f"📄 PDFs: {pdf_count}")
            st.success(f"🖼️ Images: {image_count}")
        else:
            st.warning("Data directory not found")
        
        st.divider()
        
        # Model Info
        st.subheader("🧠 Models")
        st.info("""
        - **Intent**: Custom ML Model
        - **Embeddings**: Sentence Transformers
        - **VLM**: Florence-2
        - **LLM**: OpenAI GPT-4o-mini
        """)
        
        st.divider()
        
        # Intent Types
        st.subheader("🎯 Intent Types")
        st.write("**fact**: Direct factual lookup")
        st.write("**analysis**: Comparative reasoning")
        st.write("**summary**: Summarize sections")
        st.write("**visual**: Chart/image queries")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Query Settings")
        top_k = st.slider("Top K Results", 3, 10, 5)
        max_tokens = st.slider("Max Tokens", 256, 1024, 512)
    
    # Initialize agents
    controller, retrieval_agent, vision_agent = initialize_agents()
    
    # Data ingestion status
    if 'data_ingested' not in st.session_state:
        with st.spinner("Ingesting data from 'data/' directory..."):
            try:
                # Pass vision_agent to enable image captioning during PDF ingestion
                retrieval_agent.ingest_pdfs(data_dir="data", vision_agent=vision_agent)
                retrieval_agent.ingest_images_folder(images_dir=os.path.join("data", "images"))
                st.session_state.data_ingested = True
                st.success("✅ Data ingestion complete!")
                time.sleep(1)
            except Exception as e:
                st.error(f"Error during data ingestion: {e}")
    
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
        with st.spinner("Processing query..."):
            try:
                # Handle query through controller
                res = controller.handle_query(query, top_k=top_k, max_tokens=max_tokens)
                
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
                    st.info(f"**{intent_desc}**: The query has been classified as a {intent} type question.")
                
                # Answer
                st.markdown("---")
                st.header("💡 Answer")
                answer = res.get("answer", "No answer generated.")
                st.markdown(f'<div style="padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; border-left: 4px solid #1f77b4; color: #000000;">{answer}</div>', unsafe_allow_html=True)
                
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
                        
                        with st.expander(f"📄 Result {i+1}: {source} (Page {page}) - {doc_type.upper()}", expanded=(i==0)):
                            st.write(f"**Type**: {doc_type}")
                            st.write(f"**Source**: {source}")
                            st.write(f"**Page**: {page}")
                            st.write("**Content**:")
                            st.write(document[:500] + "..." if len(document) > 500 else document)
                else:
                    st.warning("No documents retrieved.")
                
                # Image Contexts
                images = res.get("images", [])
                if images:
                    st.markdown("---")
                    st.header("🖼️ Image Contexts")
                    
                    for idx, im in enumerate(images):
                        meta = im.get("meta", {})
                        img_name = meta.get("img_name", "Unknown")
                        img_path = meta.get("img_path", "")
                        caption = im.get("caption", "")
                        ocr_text = im.get("ocr_text", "")
                        tags = im.get("tags", "")
                        
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=img_name, use_container_width=True)
                            else:
                                st.info(f"Image: {img_name}")
                        
                        with col2:
                            st.subheader(f"Image {idx + 1}: {img_name}")
                            if caption:
                                st.write("**Caption:**")
                                st.write(caption)
                            if ocr_text:
                                st.write("**OCR Text:**")
                                st.code(ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text)
                            if tags:
                                st.write("**Tags:**")
                                st.write(tags)
                        
                        if idx < len(images) - 1:
                            st.divider()
                
                # Agent Pipeline Visualization
                st.markdown("---")
                st.header("🔄 Agent Pipeline")
                st.info("""
                **Query Flow:**
                1. **Intent Agent** → Classified query as `{}`
                2. **Retrieval Agent** → Retrieved {} document(s)
                3. **Vision Agent** → Processed {} image(s)
                4. **Reasoning Agent** → Generated answer using OpenAI
                5. **Controller Agent** → Orchestrated the entire pipeline
                """.format(intent, len(retrieved), len(images)))
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.exception(e)
    
    # Example queries
    elif not submit_button:
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
            <p>Multi-Agent, Multimodal RAG System | Built with Streamlit</p>
            <p>Components: Intent Classification | RAG | VLM | LLM Reasoning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
