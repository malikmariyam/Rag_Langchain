from langchain_openai import ChatOpenAI

# Step 1: Create LLM 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 2: Define question
question = "What are the principles of quality in primary healthcare?"

# Step 3: Send prompt  to the model
response = llm.invoke(question)

# Step 4: Print result
print("🤖", response.content)
