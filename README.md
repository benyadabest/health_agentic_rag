**Zvoz Health Agentic RAG**

**Description**: This project focuses on creating a **real-time, agentic retrieval-augmented generation (RAG) system for patient education and healthcare queries**. The system integrates dynamic scraping and pre-cached FAQs to deliver actionable insights on:

- Brain cancer research.
- Clinical trials.
- Patient education materials.
- Community insights.

**Specs**:
- Real-time retrieval of live data from sources using firecrawl.dev or APIs
- Pre-cached FAQs with fallback to dynamic agents
- Agents: Research Agent fetches and processes PubMed articles, ClinicalTrial Agent queries clinicaltrials.gov for trial updates, CommunityInsight Agent fetches and processes relevant Reddit threads to capture real-world experiences, MyHealth Agent captures user's personal health context
- Patient-friendly summarization 
- Multi-source aggregation, fall-back search, contextual memory, and health document(s) upload

**Stack**:
- FastAPI, Redis, LangChain, CrewAI, firecrawl.dev
- MedLlama3-v20, BioGPT, chatgpt-4o, openAI text embeddings
- Streamlit
- Google Cloud, Docker
