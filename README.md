# CrewAI Medical Agentic RAG :sunrise_over_mountains::hospital:
[Medical Crew Chat Link](medical-crew-chat.streamlit.app)

## Description

This project focuses on creating a **real-time, agentic retrieval-augmented generation (RAG) system for patient education and health queries**. The system integrates dynamic scraping and APIs to deliver actionable insights on:

- Trusted health/medical web sources (Tavily)
- Patient education materials (Tavily)
- Community insights (Reddit via Exa)
- Medical Records with Document Analysis
    - Medical document summaries
    - Key medical topics extraction
    - Topic-specific document discussions
    - Document preview and management

## Specs
- Real-time retrieval of live data from sources using Tavily
- **NotebookLM-style source (document) features** with topic-specific contextual retrieval and summarization
- Agents: 
    1. Search Agent searches with an enhanced query and retrieves relevant answers from trusted sources
    2. CommunityInsight Agent fetches and processes relevant Reddit threads to capture real-world experiences
    3. Health Agent captures user's personal health documents as context
    4. Synthesis Agent synthesizes data from multiple sources to create clear and concise answers for the user
    5. Classifier Agent determines whether user's query is medical related, follow-up, meta, or unrelated
    6. Response Modifier Agent handles follow-up requests and questions using prev. responses
    7. Meta Query Agent handles responses to questions about the chat's capabilities, purpose, and more.
    8. Document Analyzer Agent processes and extracts insights from medical documents
- Patient-friendly summarization 
- Multi-source aggregation, basic short-term contextual memory, and health document(s) upload
- **CrewAI dynamic tool for RAG over Multiple Documents** while treating them as separate knowledge sources

## Stack
- LLM Integration: CrewAI (chatgpt-4o), Groq (llama-3.1-8b-instant)
- Agent Orchestration: Crew AI
- Framework: Streamlit
- Search: Tavily, Exa (tested: Exa, Spider, JinaAi, Serper, Tavily on speed and response quality)
- Doc Processing: PyPDF2
- Dev Tools: Pydantic

## Sample Questions
- *upload post-visit summary pdf* How does my prescribed medication compare to others on Reddit for my condition?
- How is diabetes diagnosed, and what are the latest treatments?
- *upload post-visit summary pdf* What insights can you extract from this medical record about my condition?
- What are other cancer patients saying about alternative therapies on Reddit?
- *click on extracted topic* Tell me more about [medical topic] from my document

## Screenshots
![Medical Crew Chat](chat_page.png)
![Document NotebookLM Page](doc_page.png)

## References
https://github.com/zinyando/crewai_tavily_tool_demo/blob/main/src/crewai_tavily_tool_demo/crew.py code for custom Tavily tool
