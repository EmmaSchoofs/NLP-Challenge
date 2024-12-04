import streamlit as st
import random
import time
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from tools import tool_functions



task_values = []

@CrewBase
class PersonalizedLearningAssistant():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[
                tool_functions["GroqLLMTool"](),  # Initialize GroqLLMTool
                tool_functions["PDFExtractionTool"](),  # Initialize PDFExtractionTool
            ],
            verbose=True
        )

    
    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            tools=[
                tool_functions["MarkdownFormatter"](),  # Initialize MarkdownFormatter
                tool_functions["SummaryTool"](tool_functions["GroqLLMTool"]()),  # Pass GroqLLMTool to SummaryTool
            ],
            verbose=True
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            tools=[
                tool_functions["GroqLLMTool"](),
                tool_functions["PDFExtractionTool"]()
            ],
        )

        
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            tools=[
                tool_functions["MarkdownFormatter"](),
                tool_functions["SummaryTool"](tool_functions["GroqLLMTool"]())  # Pass GroqLLMTool to SummaryTool
            ],
            output_file='report.md',  # If necessary, keep this line
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True,
            process=Process.sequential,
        )



def run_crewai_app():
    st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.text_input("ask me anything")

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
                    crew_result = crew_instance.kickoff(inputs={"query": query})


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
    run_crewai_app()