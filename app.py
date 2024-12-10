import streamlit as st
import random
import time
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from tools import tool_functions
import logging
from langchain_groq import ChatGroq
import os
from crewai_tools import PDFSearchTool
import os

# llm=ChatGroq(temperature=0,
#              model_name="llama-3.1-70b-versatile",
#              api_key=os.getenv("GROQ_API_KEY"))

task_values = []
# pdfsearchtool = PDFSearchTool(pdf="Nexus_review.pdf")
file_path = './test.pdf'

@CrewBase
class PersonalizedLearningAssistant():
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"2Registered tools: {tool_functions}")
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'


    def __init__(self):
        self.researcher_content = None


    @agent
    def researcher(self) -> Agent:
        pdf_tool = tool_functions["PDF Extraction Tool"](pdf_path=file_path)
        extracted_content = pdf_tool._run()
        self.researcher_content = extracted_content
        return Agent(
            llm=llm,
            config=self.agents_config['researcher'],
            tools=[pdf_tool],
            verbose=True
        )

    
    @agent
    def reporting_analyst(self) -> Agent:
        extracted_content = self.researcher_content
        print(extracted_content)
        return Agent(
            llm=llm,
            config=self.agents_config['reporting_analyst'],
            tools= [
            tool_functions["Markdown Formatter"](content=extracted_content),
            tool_functions["Summary Tool"](llm_tool=llm, content=extracted_content)
            ],
            # args_schema={"content": "Default content for testing"},
            verbose=True
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agents=[self.researcher()],
            tools=[
                tool_functions["PDF Extraction Tool"](pdf_path=file_path)
            ],
        )

        
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            agents=[self.reporting_analyst()],
            tools=[
                tool_functions["Markdown Formatter"](),
                tool_functions["Summary Tool"](llm_tool=llm)  
            ],
            output_file='report.md',
        )


    @crew
    def crew(self) -> Crew:
        logging.debug("Initializing Crew with tasks and agents...")
        crew = Crew(
            agents=[self.researcher(), self.reporting_analyst()],
            tasks=[self.research_task(), self.reporting_task()],
            verbose=True
        )
        logging.debug(f"Crew initialized: {crew}")
        return crew




def run_crewai_app():
    st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    content = st.text_input("ask me anything")

    if st.button("Go"):
        # Placeholder for stopwatch
            stopwatch_placeholder = st.empty()
            
            # Start the stopwatch
            start_time = time.time()
            with st.expander("Processing!"):
                # sys.stdout = StreamToExpander(st)
                with st.spinner("Generating Results"):
                    pla_instance = PersonalizedLearningAssistant()

                    crew_instance = pla_instance.crew()
                    crew_result = crew_instance.kickoff(inputs={"topic": content})


            # Stop the stopwatch
            end_time = time.time()
            total_time = end_time - start_time
            stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

            st.header("Tasks:")
            st.table({"Tasks" : task_values})

            st.header("Results:")
            st.markdown(crew_result)
# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         assistant_response = random.choice(
#             [
#                 "Hello there! How can I assist you today?",
#                 "Hi, human! Is there anything I can help you with?",
#                 "Do you need help?",
#             ]
#         )
#         # Simulate stream of response with milliseconds delay
#         for chunk in assistant_response.split():
#             full_response += chunk + " "
#             time.sleep(0.05)
#             # Add a blinking cursor to simulate typing
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    llm=ChatGroq(temperature=0,
             model_name="groq/llama3-8b-8192",
             api_key=os.getenv("GROQ_API_KEY"))
    
    run_crewai_app()