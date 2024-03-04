from langchain_openai import ChatOpenAI
import streamlit as st
import os
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig

global openai_api_key
st.set_page_config(page_title="General Web Chatbot", layout="wide")
st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 General Web Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    st.success("API key loaded from .env", icon="🚀")
else:
    openai_api_key = st.sidebar.text_input(
        label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password", key="openai_api_key"
        )
    if openai_api_key:
        st.sidebar.success("API key loaded", icon="🚀")

    os.environ["OPENAI_API_KEY"] = openai_api_key


if openai_api_key:
    # Execute the home page function
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k","gpt-3.5-turbo-1106","gpt-4-1106-preview"]

    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.9
    TEMPERATURE_STEP = 0.01
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    # freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
    temperature = st.sidebar.slider(
                    label="Temperature",
                    min_value=TEMPERATURE_MIN_VALUE,
                    max_value=TEMPERATURE_MAX_VALUE,
                    value=TEMPERATURE_DEFAULT_VALUE,
                    step=TEMPERATURE_STEP,)
    
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}
        
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)
    
    s_prompt = st.sidebar.selectbox(label="Prompt Selection", options=["Default Prompt", "Custom Prompt"])
    if s_prompt == "Default Prompt":
        sys_prompt = """Assistant is a large language model for helpful information giving QA System.

                        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand."""
    elif s_prompt == "Custom Prompt":
        sys_prompt = st.sidebar.text_area("Custom Prompt", placeholder="Enter your custom prompt here")
    
    if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(model_name=model_name, temperature=temperature, top_p=top_p, streaming=True)
        tools = [DuckDuckGoSearchRun(name="Search")]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, 
                                                                tools=tools,
                                                                system_message=sys_prompt,
                                                                verbose=True,)
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            cfg = RunnableConfig()
            cfg["callbacks"] = [st_cb]
            response = executor.invoke(prompt, cfg)
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]

else:
    st.info("Please add your OpenAI API key to continue.")