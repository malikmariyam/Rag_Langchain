from tools.rag_tool import qa_from_pdf
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI


# Step 1: LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 2: Register the tool
tools = [qa_from_pdf]

# Step 3: Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 4: Ask a question from the PDF
response = agent.run("what is the importance of Community engagement?")
print("Answer:", response)
