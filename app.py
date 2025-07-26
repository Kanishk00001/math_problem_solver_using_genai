import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit app config
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver Using Google Gemma 2")

# Groq API Key input
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Searches Wikipedia for information about any topic."
)

# Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solves mathematical expressions and provides answers with explanation."
)

# Reasoning Prompt Template
prompt = """
You are an agent tasked with solving user's mathematical questions. 
Logically arrive at the solution and provide a detailed explanation, displaying it pointwise.

Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Reasoning Tool
reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Answers logic-based and reasoning math questions."
)

# Agent Initialization
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Session state for message history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a math chatbot who can answer all your math problems!"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input area
question = st.text_area("Enter your question:", "Sum of fruits if I have 2 banana and 1 apple")

# Button logic
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Callback Handler
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Run the assistant agent
            try:
                response = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
                st.success("Response generated successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
