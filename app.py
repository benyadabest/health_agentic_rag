__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Any, Type
from crewai.tools import BaseTool
from enum import Enum
from PyPDF2 import PdfReader
from pathlib import Path
from exa_py import Exa
from crewai.tools import tool

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
def init_agents():
    txt_tool = load_text_tool()
    reddit_tool = load_reddit_tool()
    tavily_tool = load_tavily_tool()
    llm = init_llm()
    
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
        llm=LLM(model="gpt-4o", temperature=0),
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
        llm=LLM(model="gpt-4o", temperature=0),
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
        llm=LLM(model="gpt-4o", temperature=0),
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

def handle_meta_query(query: str) -> str:

    meta_llm = LLM(model="gpt-4o", temperature=0.4)
    system_prompt = """
    You are a specialized medical assistant. You handle meta-queries by explaining:
    - Your purpose: to assist with health-related questions, document summaries, and personalized insights.
    - Your strengths: trusted medical sources, Reddit integration, and document analysis.
    - How you differ from general-purpose AIs like ChatGPT.
    """
    prompt = f"{system_prompt}\n\nUser query: {query}\n\nResponse:"
    
    response = meta_llm.generate(prompt)
    return response

def create_crew_tasks(query, agents, qa_pairs=None):
    health_agent, search_agent, community_agent, synthesis_agent = agents
    
    health_context_task = Task(
        description=(
            f"Check if the query {query} EXPLICITLY references the user's health "
            "(e.g., contains 'my condition,' 'for me') "
            "If NO, then return '' and ignore the rest of task. If YES, proceed to the following steps: "
            "Using the tool, Provide relevant medical context to help answer the query "
            "Ensure there's no hallucination and include citations for each piece of information"
        ),
        expected_output="A structured health context report.",
        agent=health_agent,
        async_execution=True
    )

    search_task = Task(
        description=(
            f"Search websites for information about: {query}\n"
            "Use the tool to retrieve information from the web "
            "Your task: "
            "1. Prefix the query with: (site:.gov OR site:.edu OR site:.org) and Search\n"
            "3. Return your results"
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
        f"Create a comprehensive answer for: {query}\n"
        "Using:\n"
        "- Health Context (if available)\n"
        "- Web Search Results (if applicable)\n"
        "- Reddit Search Results (if available)\n"
    )
    
    ##created by claude.ai
    if qa_pairs:
        synthesis_desc += "\nPrevious Conversation Context (Most Recent First):\n"
        for i, qa in enumerate(qa_pairs, 1):
            synthesis_desc += (
                f"\nQ{i}: {qa['question']}\n"
                f"A{i}: {qa['answer']}\n"
            )
        
        synthesis_desc += (
            "\nUse this previous context when relevant to the current query. "
            "If the current query references or relates to previous answers, "
            "incorporate that information appropriately."
        )

    synthesis_desc += (
        "\nRequirements:\n"
        "If reddit Search Results are included, your response should include primarily this content\n"
        "- List ALL included sources at the end\n"
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

def handle_pdf_upload():
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Drag and drop medical documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to include in the context"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            with st.expander("View Uploaded Files"):
                for file in uploaded_files:
                    st.write(f"📎 {file.name}")
            
            # Process files
            all_text = []
            for file in uploaded_files:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                all_text.append(text)
            
            combined_text = "\n\n".join(all_text)
            with open("health_data.txt", "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            st.session_state.document_content = combined_text
            st.cache_resource.clear()
            
            if st.checkbox("Show extracted text"):
                st.text_area("Content", combined_text, height=300)
            
            st.success("✅ Content saved to health_data.txt")
            return True
            
    return False

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! We are your medical team aiming to answer your questions!"}
        ]
    if "document_content" not in st.session_state:
        st.session_state.document_content = ""
    
    has_uploads = handle_pdf_upload()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    agents = init_agents()
    llm = init_llm()
    
    if prompt := st.chat_input(
        "What would you like to know?"):
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
                        return handle_meta_query(prompt)
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