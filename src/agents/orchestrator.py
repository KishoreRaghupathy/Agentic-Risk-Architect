import os
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# Define Structured Output for the Final Advisory
class RiskAdvisory(BaseModel):
    """Structured output for the final risk advisory."""
    risk_score: int # 1-10
    market_sentiment: str
    technical_drivers: list[str]
    action_plan: str

class RiskOrchestrator:
    """Orchestrates the multi-agent AI committee for risk assessment."""
    
    def __init__(self, model_data: dict, news_headlines: list):
        self.model_data = model_data # From Day 2
        self.news_headlines = news_headlines # From Day 1
        
        # Configure LLM backend
        self.llm = self._initialize_llm()
        self.setup_agents()
        
    def _initialize_llm(self):
        """Initializes the LLM backend based on available API keys."""
        if os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI:
            # Using Gemini backend as one of the preferred options
            return ChatGoogleGenerativeAI(model="gemini-3.1-pro")
        elif os.getenv("OPENAI_API_KEY") and ChatOpenAI:
            # Fallback/Alternative Open AI backend
            return ChatOpenAI(model="gpt-4-turbo")
        return None # Falls back to CrewAI's default if no specific backend is configured
        
    def setup_agents(self):
        """Initializes the three AI personas."""
        # 1. Qualitative Agent
        self.sentiment_analyst = Agent(
            role='Senior Market Sentiment Analyst',
            goal='Analyze news for hidden financial risks',
            backstory="""You are an expert in financial NLP. You look beyond 
            headlines to find market fear or systemic risks.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # 2. Quantitative Agent
        self.quant_auditor = Agent(
            role='Lead Quantitative Auditor',
            goal='Explain ML model outputs to non-technical stakeholders',
            backstory="""You specialize in XAI (Explainable AI). You turn 
            SHAP values and XGBoost probabilities into narrative logic.""",
            verbose=True,
            llm=self.llm
        )

        # 3. The Decision Agent
        self.cro = Agent(
            role='Chief Risk Officer',
            goal='Synthesize all data into a final risk advisory',
            backstory="""You are the final gatekeeper. You balance the news 
            and the numbers to issue high-stakes trade advisories.""",
            verbose=True,
            llm=self.llm
        )

    def run_sprint(self):
        """Executes the sequential multi-agent workflow."""
        # Task 1: Analyze News
        t1 = Task(
            description=f"Analyze these headlines: {self.news_headlines}. Identify the top 3 sentiment drivers.",
            agent=self.sentiment_analyst,
            expected_output="A 3-point bulleted list of market sentiment drivers."
        )

        # Task 2: Audit ML Model
        t2 = Task(
            description=f"Interpret these model results: {self.model_data}. Focus on the SHAP values.",
            agent=self.quant_auditor,
            expected_output="An explanation of the top 3 technical drivers of risk."
        )

        # Task 3: Final Advisory
        t3 = Task(
            description="""Review the Sentiment report and the Quant report. 
            Assign a final Risk Score (1-10) and an Action Plan.""",
            agent=self.cro,
            output_json=RiskAdvisory,
            expected_output="A structured Risk Advisory object."
        )

        crew = Crew(
            agents=[self.sentiment_analyst, self.quant_auditor, self.cro],
            tasks=[t1, t2, t3],
            process=Process.sequential
        )
        
        return crew.kickoff()
