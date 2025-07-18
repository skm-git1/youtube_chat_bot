import streamlit as st
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# Import for different LLM providers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         color: #FF6B6B;
#         font-size: 3rem;
#         font-weight: bold;
#         margin-bottom: 2rem;
#     }
    
#     .sub-header {
#         text-align: center;
#         color: #4ECDC4;
#         font-size: 1.2rem;
#         margin-bottom: 3rem;
#     }
    
#     .video-container {
#         display: flex;
#         justify-content: center;
#         margin: 2rem 0;
#     }
    
#     .chat-container {
#         background-color: #f8f9fa;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 4px solid #4ECDC4;
#         margin: 1rem 0;
#     }
    
#     .question-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #2196f3;
#     }
    
#     .answer-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #4caf50;
#     }
    
#     .error-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #f44336;
#     }
    
#     .success-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #4caf50;
#     }
    
#     .warning-box {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #ff9800;
#     }
    
#     .provider-info {
#         background-color: #f8f9fa;
#         padding: 0.5rem;
#         border-radius: 5px;
#         margin: 0.5rem 0;
#         font-size: 0.9rem;
#     }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
    /* Base font color for all containers */
    .question-box,
    .answer-box,
    .error-box,
    .success-box,
    .warning-box,
    .chat-container,
    .provider-info {
        color: var(--textColor);
        padding: 1rem; 
        background-color: var(--secondaryBackgroundColor);
    }

    /* Accent borders */
    .chat-container     { border-left: 4px solid var(--primaryColor); }
    .question-box       { border-left: 4px solid #2196f3; }  /* you can swap for any accent */
    .answer-box         { border-left: 4px solid #4caf50; }
    .error-box          { border-left: 4px solid #f44336; }
    .success-box        { border-left: 4px solid #4caf50; }
    .warning-box        { border-left: 4px solid #ff9800; }
    .provider-info      { border-left: none; }

    /* Headers use theme primary text */
    .main-header { 
        text-align: center;
        color: var(--primaryColor);
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: var(--textColor);
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    /* Video & chat layout */
    .video-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcript_processed' not in st.session_state:
    st.session_state.transcript_processed = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'main_chain' not in st.session_state:
    st.session_state.main_chain = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def process_transcript(transcript):
    """Process transcript and create retriever"""
    try:
        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        return retriever
    except Exception as e:
        st.error(f"Error processing transcript: {str(e)}")
        return None

def create_llm(provider, api_key, model_name=None, hf_token=None):
    """Create LLM based on provider"""
    try:
        if provider == "OpenAI":
            if not api_key:
                st.error("OpenAI API key is required")
                return None
            return ChatOpenAI(
                model=model_name or "gpt-4o-mini", 
                temperature=0.2, 
                api_key=api_key
            )
        
        elif provider == "Google Gemini":
            if not GEMINI_AVAILABLE:
                st.error("Google Gemini dependencies not installed. Please install: pip install langchain-google-genai")
                return None
            if not api_key:
                st.error("Google Gemini API key is required")
                return None
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-pro",
                temperature=0.2,
                google_api_key=api_key
            )
        
        elif provider == "Hugging Face":
            if not HUGGINGFACE_AVAILABLE:
                st.error("Hugging Face dependencies not installed. Please install: pip install langchain-huggingface transformers torch")
                return None
            
            model_id = model_name or "microsoft/DialoGPT-medium"
            
            # For Hugging Face, we'll use a pipeline approach
            try:
                if hf_token:
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                
                hf_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    tokenizer=model_id,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    trust_remote_code=True
                )
                
                return HuggingFacePipeline(pipeline=hf_pipeline)
            except Exception as e:
                st.error(f"Error loading Hugging Face model: {str(e)}")
                return None
        
        else:
            st.error(f"Unknown provider: {provider}")
            return None
            
    except Exception as e:
        st.error(f"Error creating LLM: {str(e)}")
        return None

def create_qa_chain(retriever, provider, api_key, model_name=None, hf_token=None):
    """Create the Q&A chain"""
    try:
        # Initialize LLM
        llm = create_llm(provider, api_key, model_name, hf_token)
        if not llm:
            return None
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant that answers questions about YouTube videos.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            Be concise but informative in your responses.

            Context: {context}
            Question: {question}
            
            Answer:
            """,
            input_variables=['context', 'question']
        )
        
        # Create formatting function
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        
        # Create parallel chain
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        # Create main chain
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        
        return main_chain
    except Exception as e:
        st.error(f"Error creating Q&A chain: {str(e)}")
        return None

# Main UI
st.markdown('<h1 class="main-header">🎥 YouTube Video Q&A</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about any YouTube video with AI-powered answers</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider selection
    available_providers = ["OpenAI"]
    if GEMINI_AVAILABLE:
        available_providers.append("Google Gemini")
    if HUGGINGFACE_AVAILABLE:
        available_providers.append("Hugging Face")
    
    provider = st.selectbox(
        "Select AI Provider",
        available_providers,
        help="Choose which AI provider to use for generating answers"
    )
    
    # API Key inputs based on provider
    api_key = None
    model_name = None
    hf_token = None
    
    if provider == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        model_name = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
            help="Select OpenAI model"
        )
        st.markdown('<div class="provider-info">💡 Get your API key from: https://platform.openai.com/api-keys</div>', unsafe_allow_html=True)
    
    elif provider == "Google Gemini":
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        model_name = st.selectbox(
            "Model",
            ["gemini-pro", "gemini-pro-vision"],
            help="Select Gemini model"
        )
        st.markdown('<div class="provider-info">💡 Get your API key from: https://makersuite.google.com/app/apikey</div>', unsafe_allow_html=True)
    
    elif provider == "Hugging Face":
        hf_token = st.text_input(
            "Hugging Face Token (Optional)",
            type="password",
            help="Enter your Hugging Face token for private models"
        )
        model_name = st.selectbox(
            "Model",
            [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill",
                "google/flan-t5-base",
                "google/flan-t5-large"
            ],
            help="Select Hugging Face model"
        )
        st.markdown('<div class="provider-info">💡 Get your token from: https://huggingface.co/settings/tokens</div>', unsafe_allow_html=True)
        st.markdown('<div class="warning-box">⚠️ Hugging Face models may be slower and require more resources</div>', unsafe_allow_html=True)
    
    if not api_key and provider != "Hugging Face":
        st.warning(f"Please enter your {provider} API key to continue.")
    
    st.markdown("---")
    st.markdown("### 📝 How to use:")
    st.markdown("1. Select your AI provider")
    st.markdown("2. Enter the required API key/token")
    st.markdown("3. Paste a YouTube video URL")
    st.markdown("4. Click 'Process Video'")
    st.markdown("5. Ask questions about the video")
    
    st.markdown("---")
    st.markdown("### 🔍 Example questions:")
    st.markdown("- What is this video about?")
    st.markdown("- Summarize the main points.")
    st.markdown("- What does the speaker say about...?")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Video URL input
    st.subheader("📹 Enter YouTube Video URL")
    video_url = st.text_input(
        "Paste YouTube URL here:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Process video button
    can_process = (provider == "Hugging Face") or (api_key is not None)
    
    if st.button("🔄 Process Video", type="primary", disabled=not can_process):
        if video_url:
            video_id = extract_video_id(video_url)
            
            if video_id:
                with st.spinner("Processing video transcript..."):
                    # Get transcript
                    transcript = get_video_transcript(video_id)
                    
                    if transcript:
                        # Process transcript
                        retriever = process_transcript(transcript)
                        
                        if retriever:
                            # Create Q&A chain
                            main_chain = create_qa_chain(retriever, provider, api_key, model_name, hf_token)
                            
                            if main_chain:
                                # Store in session state
                                st.session_state.retriever = retriever
                                st.session_state.main_chain = main_chain
                                st.session_state.transcript_processed = True
                                st.session_state.video_title = f"Video ID: {video_id}"
                                st.session_state.chat_history = []
                                st.session_state.current_provider = provider
                                
                                st.markdown(f'<div class="success-box">✅ Video processed successfully using {provider}! You can now ask questions.</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-box">❌ Failed to create Q&A system.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="error-box">❌ Failed to process transcript.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">❌ No transcript available for this video. Please try a different video.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ Invalid YouTube URL. Please check the URL and try again.</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a YouTube video URL.")

with col2:
    # Video preview
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            st.subheader("📺 Video Preview")
            st.video(f"https://www.youtube.com/watch?v={video_id}")

# Chat interface
if st.session_state.transcript_processed:
    st.markdown("---")
    
    # Show current provider
    current_provider = getattr(st.session_state, 'current_provider', 'Unknown')
    st.markdown(f"### 💬 Ask Questions About the Video (Using {current_provider})")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f'<div class="question-box"><strong>Q:</strong> {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><strong>A:</strong> {answer}</div>', unsafe_allow_html=True)
    
    # Question input
    question = st.text_input("Ask a question about the video:", key="question_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and question:
        if st.session_state.main_chain:
            with st.spinner("Generating answer..."):
                try:
                    answer = st.session_state.main_chain.invoke(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display the new Q&A
                    st.markdown(f'<div class="question-box"><strong>Q:</strong> {question}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-box"><strong>A:</strong> {answer}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.error("Please process a video first.")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>Built with ❤️ using Streamlit, LangChain, and multiple AI providers</p>
        <p>Supported providers: OpenAI{', Google Gemini' if GEMINI_AVAILABLE else ''}{', Hugging Face' if HUGGINGFACE_AVAILABLE else ''}</p>
        <p>Make sure the YouTube video has captions/subtitles available for processing.</p>
    </div>
    """,
    unsafe_allow_html=True
)