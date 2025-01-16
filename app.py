# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from crewai import Agent, Task, Crew, LLM, Knowledge
from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Any, Type
from crewai.tools import BaseTool
from enum import Enum
import json
from PyPDF2 import PdfReader
from pathlib import Path
from exa_py import Exa
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

load_dotenv()

st.set_page_config(
    page_title="Medical Crew Chat",
    page_icon="🏥"
)
st.title("Medical Crew Chat :mostly_sunny::hospital:")

class TavilySearchInput(BaseModel):
    query: Annotated[str, Field(description="The search query string")]
    max_results: Annotated[
        int, Field(description="Maximum number of results to return", ge=1, le=10)
    ] = 5
    search_depth: Annotated[
        str,
        Field(
            description="Search depth: 'basic' or 'advanced'",
            choices=["basic", "advanced"],
        ),
    ] = "basic"


class TavilySearchTool(BaseTool):
    name: str = "Tavily Search"
    description: str = (
        "Use the Tavily API to perform a web search and get AI-curated results."
    )
    args_schema: Type[BaseModel] = TavilySearchInput
    client: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.client = TavilyClient(api_key=api_key)

    def _run(self, query: str, max_results=5, search_depth="basic") -> str:
        if not self.client.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        try:
            response = self.client.search(
                query=query, max_results=max_results, search_depth=search_depth
            )
            return self._process_response(response)
        except Exception as e:
            return f"An error occurred while performing the search: {str(e)}"

    def _process_response(self, response: dict) -> str:
        if not response.get("results"):
            return "No results found."

        results = []
        for item in response["results"][:5]:
            title = item.get("title", "No title")
            content = item.get("content", "No content available")
            url = item.get("url", "No URL available")
            results.append(f"Title: {title}\nContent: {content}\nURL: {url}\n")

        return "\n".join(results)


exa_api_key = os.getenv("EXA_API_KEY")

@tool("Exa Reddit search")
def search_reddit_posts(query: str, num_results: int = 5) -> str:
    """"
    Search Reddit posts using Exa's semantic search capabilities.
    
    Args:
        query (str): The search query.
        num_results (int): Number of results to retrieve.
    
    Returns:
        str: Formatted string containing the Reddit post details.
    """
    exa = Exa(exa_api_key)

    search_query = f"site:reddit.com {query}"

    response = exa.search_and_contents(
        search_query,
        type="neural",          
        use_autoprompt=True,   
        num_results=num_results,
        include_domains=["Reddit.com"],
        highlights=True
    )

    if not response.results:
        return "No results found."

    parsed_results = ''.join([
        f"<Title id={idx}>{result.title}</Title>\n"
        f"<URL id={idx}>{result.url}</URL>\n"
        f"<Highlight id={idx}>{' '.join(result.highlights)}</Highlight>\n\n"
        for idx, result in enumerate(response.results)
    ])

    return parsed_results

reddit_tool = search_reddit_posts

@st.cache_resource
def load_text_tool():
    with open("health_data.txt", "r") as f:
        content = f.read()
    return TXTSearchTool(txt=content)

@st.cache_resource
def load_tavily_tool():
    return TavilySearchTool()

@st.cache_resource
def load_reddit_tool():
    return reddit_tool

@st.cache_resource
def init_llm():
    return LLM("groq/llama-3.1-8b-instant", temperature=0)

@st.cache_resource
def init_llm2():
    return LLM(model="gpt-4o", temperature=0)

@st.cache_resource
def init_agents():
    txt_tool = load_text_tool()
    reddit_tool = load_reddit_tool()
    tavily_tool = load_tavily_tool()
    llm = init_llm()
    llm2 = init_llm2()
    
    health_agent = Agent(
        role="Smart Health Context Analyzer",
        goal="Determine if query is user-specific and if necessary provide user medical context",
        tools=[txt_tool],
        verbose=True,
        backstory=(
            "You analyze patient data "
            "to determine if any relevant information exists for the query. "
            "However, only do so if the query is user-specific (e.g., contains 'my condition,' 'for me')"
        ),
        llm=llm2,
    )

    search_agent = Agent(
        role="Senior Web Search Researcher",
        goal="Search trusted medical websites for relevant information for the query.",
        tools=[tavily_tool], 
        verbose=True,
        backstory=(
            "You are skilled at finding reliable medical information from trusted websites "
            "that end with .gov, .edu, and .org. "
            "An expert web researcher that uses the web extremely well"
        ),
        llm=llm
    )

    community_agent = Agent(
        role="Senior Reddit Researcher",
        goal="Use Exa Reddit Search for relevant information for the query.",
        tools = [reddit_tool],
        verbose=True,
        backstory=(
            "You are skilled at finding reliable medical information from Reddit.com, ONLY search this domain"
        ),
        llm=llm2,
    )

    synthesis_agent = Agent(
        role="Professional Medical Information Synthesizer",
        goal="Combine all available information into a coherent response",
        tools=[],
        verbose=True,
        backstory=(
            "You synthesize data from multiple sources to create clear and concise answers for the user."
            "When given previous conversation context, you incorporate it intelligently into your responses if needed."
        ),
        llm=llm2,
    )
    
    return health_agent, search_agent, community_agent, synthesis_agent

#created by claude.ai
def get_recent_qa_pairs(max_pairs=3):
    qa_pairs = []
    messages = st.session_state.messages[1:]  
    i = len(messages) - 1
    
    while i > 0 and len(qa_pairs) < max_pairs:
        if i > 0:
            current = messages[i]
            previous = messages[i-1]
            
            if current["role"] == "assistant" and previous["role"] == "user":
                qa_pairs.append({
                    "question": previous["content"],
                    "answer": current["content"]
                })
            i -= 1
    
    return qa_pairs



def create_crew_tasks(query, agents, qa_pairs=None, doc_context=None):
    health_agent, search_agent, community_agent, synthesis_agent = agents
    
    use_doc = False
    if isinstance(query, dict):
        topic = query["topic"]
        doc_context = query["content"]
        search_query = f"medical information about {topic}"
        use_doc = True
    else:
        topic = query
        search_query = query

    print(str(use_doc))

    if doc_context:
        txt_tool2 = TXTSearchTool(txt=doc_context)
        health_agent.tools = [txt_tool2]

    health_context_task = Task(
        description=(
            f"Using the tool, analyze the following:"
            f"\nQuery/Topic: {topic}"
            f"\nIs this from a document? {use_doc}"
            "\nInstructions:"
            f"\n1. If from document (use_doc=True): Provide relevant medical context about {topic} from the document"
            "\n2. If not from document: Check if query references user's health (e.g., 'my condition,' 'for me')"
            "\n   - If NO, return ''"
            "\n   - If YES, provide relevant medical context"
            "\nEnsure there's no hallucination and include citations for each piece of information"
        ),
        expected_output="A structured health context report.",
        agent=health_agent,
        async_execution=True
    )

    search_task = Task(
        description=(
            f"Search websites for information about: {search_query}\n"
            "Use the tool to retrieve information from the web "
            "Your task: "
            "1. Prefix the query with: (site:.gov OR site:.edu OR site:.org) and Search\n"
            "3. Return your results with clear citations"
        ),
        expected_output="Content from medical sources with clear citations.",
        agent=search_agent,
        async_execution=True
    )

    community_task = Task(
        description=(
            f"Check if the query {query} EXPLICITLY contains 'Reddit', 'reddit', 'others', or 'community'"
            "If NO, then return '' and ignore the rest of task. If YES, proceed to the following steps: "
            "1. Search to provide relevant information from Reddit.com\n"
            "2. Return your results"
        ),
        expected_output="Content from reddit with clear citations",
        agent=community_agent,
        async_execution=True
    )

    synthesis_desc = (
        f"Create a comprehensive, patient-friendly answer for: {query}\n"
        "Using:\n"
        "- Health Context (if available)\n"
        "- Web Search Results (if applicable)\n"
        "- Reddit Search Results (if available)\n"
        "\nRequirements:\n"
        "- If reddit Search Results are included, your response should include primarily this content\n"
        "- ALWAYS List ALL included sources at the end\n"
        "- Include a brief medical disclaimer\n"
        "- Keep the response clear and well-structured"
    )

    synthesis_task = Task(
        description=synthesis_desc,
        expected_output="A clear medical response with proper organization and citations with links.",
        context=[health_context_task, search_task, community_task],
        agent=synthesis_agent
    )
    
    return [health_context_task, search_task, community_task, synthesis_task]

class QueryType(Enum):
    MEDICAL = 1
    FOLLOWUP = 2
    META = 3
    IRRELEVANT = 4

def classify_query(query: str) -> QueryType:
    classifier_agent = Agent(
        role="Medical Query Classifier",
        goal="Classify queries as medical questions, follow-ups, or irrelevant",
        tools=[],
        verbose=True,
        backstory="You determine whether questions are new medical queries, follow-ups requests or questions, or unrelated to medicine AND not a follow up.",
        llm=LLM(model="gpt-4o", temperature=0)
    )
    
    classification_task = Task(
        description=f"""
        Classify this query: "{query}"
        
        Respond with:
        1 - If it's a new medical query that:
            - Asks about health conditions, symptoms, or treatments
            - Seeks medical advice or information 
            - Discusses healthcare or medical procedures
            - Mentions medical terms or body parts
            - Asks about health-related lifestyle changes
            - MENTIONS 'health', 'medication', 'medical', 'document', 'documents', 'reddit'

        2 - If it's a follow-up that:
            - Is a follow-up request about previous medical information (e.g., "make it simpler", "explain that better", "can you be more concise")
            - Requests clarification of previous explanation
            - Asks for simpler/more detailed/more concise version
            - References previous response
            - Asks for elaboration on topic just discussed

        3 - If it's a meta-query about the assistant:
            - Asks about the assistant’s capabilities, purpose, or comparison to other AI systems
            - Examples include: "What can you do?", "Why are you better than ChatGPT?", "Who are you?"
            - States a friendly greeting, compliment, or thank you
            - Examples include: "hi", "hello", "what's up", "thank you"
            
        4 - If it's irrelevant (not medical AND not follow-up)
        
        Only respond with the number: 1, 2, 3, or 4
        """,
        expected_output="Classification number (1, 2, 3, 4)",
        agent=classifier_agent
    )
    
    crew = Crew(
        agents=[classifier_agent],
        tasks=[classification_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return QueryType(int(str(result).strip()))

def process_query(query, agents, qa_pairs=None):
    tasks = create_crew_tasks(query, agents, qa_pairs)
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )
    
    return crew.kickoff()

def handle_follow_up(query: str, previous_response: str) -> str:
    modifier_agent = Agent(
        role="Medical Response Modifier",
        goal="Modify the previous medical response based on the follow-up request",
        tools=[],
        verbose=True,
        backstory="You help modify medical explanations based on user needs while maintaining accuracy.",
        llm=LLM(model="gpt-4o", temperature=0)
    )
    
    task = Task(
        description=f"""
        Modify this previous medical response based on the user's request:
        
        User's Follow-up: {query}
        Previous Response: {previous_response}
        
        Keep medical information accurate while adjusting the format/style as requested.
        NO adding new information. NO hallucination.
        """,
        expected_output="Modified response",
        agent=modifier_agent
    )
    
    crew = Crew(
        agents=[modifier_agent],
        tasks=[task],
        verbose=True
    )
    
    return crew.kickoff()

def analyze_document(text, file_name):
    """
    Analyze medical document using CrewAI to extract summary and key topics
    """
    from crewai import Agent, Task, Crew, LLM
    
    llm = LLM(model="gpt-4o", temperature=0)
    
    summarizer_agent = Agent(
        role="Medical Document Summarizer",
        goal="Create concise, accurate summaries of medical documents",
        backstory="""Expert at distilling complex medical documents into clear, 
        actionable summaries while preserving key medical information""",
        llm=llm
    )
    
    topic_agent = Agent(
        role="Medical Topic Extractor",
        goal="Identify key medical topics and themes from documents",
        backstory="""Specialist in identifying and categorizing medical topics, 
        conditions, treatments, and themes from healthcare documents""",
        llm=llm
    )
    
    summary_task = Task(
        description=f"""
        Create a concise summary of this medical document.
        Requirements:
        - 1-2 paragraphs maximum
        - Focus on main medical findings/information
        - Preserve any critical health data
        - Use clear, patient-friendly language
        
        Document text:
        {text}
        
        Return ONLY the summary text, no additional formatting or explanations.
        """,
        expected_output="A clear, concise medical document summary in 1-2 paragraphs.",
        agent=summarizer_agent
    )
    
    topics_task = Task(
        description=f"""
        Extract 4-7 key medical topics from this document.
        Requirements:
        - Focus on significant medical themes
        - Include conditions, treatments, or procedures mentioned
        - Identify any recurring medical concepts
        - Make topics specific enough to be meaningful
        - Format each topic as a clear phrase (2-5 words)
        
        Document text:
        {text}
        
        Return ONLY the list of topics, one per line, no numbering or bullets.
        """,
        expected_output="A list of 4-7 key medical topics from the document.",
        agent=topic_agent
    )
    
    summary_crew = Crew(
        agents=[summarizer_agent],
        tasks=[summary_task],
        verbose=True
    )
    
    topics_crew = Crew(
        agents=[topic_agent],
        tasks=[topics_task],
        verbose=True
    )
    
    try:
        summary_result = summary_crew.kickoff()
        summary = str(summary_result).strip()
        
        topics_result = topics_crew.kickoff()
        topics = [topic.strip() for topic in str(topics_result).strip().split('\n') 
                 if topic.strip()]
        
        analysis = {
            "summary": summary,
            "topics": topics,
            "file_name": file_name,
            "file_path": str(Path("documents") / file_name)
        }
        
        return analysis
        
    except Exception as e:
        st.error(f"Error processing document analysis: {str(e)}")
        return None

def save_document_analysis(file_name: str, analysis: dict):
    docs_dir = Path("documents")
    docs_dir.mkdir(exist_ok=True)
    
    summaries_path = docs_dir / "summaries.json"
    
    if summaries_path.exists():
        with open(summaries_path, "r") as f:
            summaries = json.load(f)
    else:
        summaries = {}
    
    summaries[file_name] = analysis
    
    with open(summaries_path, "w") as f:
        json.dump(summaries, f, indent=2)

def handle_document_upload():
    with st.sidebar:
        st.header("📄 Document Upload")
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}
            
        if "document_content" not in st.session_state:
            st.session_state.document_content = ""
            
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
            
        uploaded_files = st.file_uploader(
            "Upload medical documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to include in the context"
        )
        
        st.subheader("📋 Files Dashboard")
        dashboard = st.container()
        
        with dashboard:
            if st.session_state.uploaded_files:
                for file_name, data in st.session_state.uploaded_files.items():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(f"📄 {file_name}", key=f"file_{file_name}"):
                            if st.checkbox(f"Show summary for {file_name}", key=f"show_{file_name}"):
                                st.write(data["summary"])
                                st.write("Key Topics:")
                                st.write(", ".join(data["topics"]))
                    with col2:
                        if st.button("❌", key=f"delete_{file_name}"):
                            del st.session_state.uploaded_files[file_name]
                            if not st.session_state.uploaded_files:
                                st.session_state.document_content = ""
                            st.rerun()
            else:
                st.info("No files uploaded yet")
        
        if uploaded_files and not st.session_state.processing_complete:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            all_text = []
            for file in uploaded_files:
                # Skip if file already processed
                if file.name in st.session_state.uploaded_files:
                    continue
                    
                text = ""
                try:
                    reader = PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    with st.spinner(f"Analyzing {file.name}..."):
                        analysis = analyze_document(text, file.name)
                        if analysis:
                            st.session_state.uploaded_files[file.name] = {
                                "summary": analysis["summary"],
                                "topics": analysis["topics"],
                                "content": text,
                                "file_name": file.name
                            }
                            st.success(f"✅ {file.name} analyzed")
                    
                    all_text.append(text)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    continue
            
            if all_text:
                st.session_state.document_content = "\n\n".join(
                    [data["content"] for data in st.session_state.uploaded_files.values()]
                )
            
            st.session_state.processing_complete = True
            st.cache_resource.clear()
            st.rerun()
            
        if not uploaded_files:
            st.session_state.processing_complete = False
        
        return bool(st.session_state.uploaded_files)

def get_document_context(doc_name=None):
    if doc_name and doc_name in st.session_state.uploaded_files:
        return st.session_state.uploaded_files[doc_name]["content"]
    return st.session_state.document_content

def handle_meta_query(query: str) -> str:
    content="""
        I am a specialized medical chat that generates responses from multiple specialized agents. My purpose is to:
        - Assist with health-related questions, document summaries, and personalized insights.
        - Use only trusted sources and patient-friendly formats.
        - Provide insights from real-world communities like Reddit.
        - Analyze user-uploaded documents and extract summaries and key topics.
        - Offer a more targeted approach than general-purpose models like ChatGPT.
        - Unlike ChatGPT, due to modular architecture, be able to handle document upload and real-time web retrieval simultaneously.
        """
    
    string_source = StringKnowledgeSource(
        content=content,
    )
    meta_agent = Agent(
        role="Meta Query Responder",
        goal="Respond to greetings and queries about the chat's purpose, capabilities, and strengths. ",
        backstory=(
            "You are a Meta Query Responder, designed to answer questions about the chat's purpose, "
            "capabilities, and how it differs from general-purpose AIs like ChatGPT. Your responses should "
            "be clear, concise, and user-friendly, incorporating knowledge about the assistant."
            "You also are responsible for being friendly and responding to greetings, compliments, thank yous."
        ),
        verbose=True,
        llm=LLM(model="gpt-4o", temperature=0),
        knowledge_sources=[string_source]
    )

    task = Task(
        description=f"Respond to the meta-query: '{query}' using the knowledge source.",
        expected_output="A detailed explanation of the crew/chat's purpose, strengths, and differentiators. Or simple friendly response back",
        agent=meta_agent
    )

    crew = Crew(agents=[meta_agent], tasks=[task], verbose=True, knowledge_sources=[string_source])
    result = crew.kickoff()
    return result

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! We are your medical team aiming to answer your questions!"}
        ]
    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
        
    has_uploads = handle_document_upload()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    agents = init_agents()
    
    if "doc_query" in st.session_state:
        query_data = st.session_state.pop("doc_query")
        display_query = f"Discussion of '{query_data['topic']}' from document: {query_data['doc_name']}"
        
        st.session_state.messages.append({"role": "user", "content": display_query})
        with st.chat_message("user"):
            st.markdown(display_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    qa_pairs = get_recent_qa_pairs(max_pairs=3)
                    response = process_query(query_data, agents, qa_pairs)  # Pass the entire query_data
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    recent_qa = get_recent_qa_pairs(max_pairs=1)
                    query_type = classify_query(prompt)
                    
                    if query_type == QueryType.MEDICAL:
                        qa_pairs = get_recent_qa_pairs(max_pairs=3)
                        response = process_query(prompt, agents, qa_pairs)
                    
                    elif query_type == QueryType.FOLLOWUP and recent_qa:
                        response = handle_follow_up(prompt, recent_qa[0]["answer"])
                    
                    elif query_type == QueryType.META:
                        response = handle_meta_query(prompt)
                    else:
                        response = "I'm specifically designed to help with medical and health-related questions. Could you please ask a health-related question or rephrase your query to focus on medical aspects?"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()