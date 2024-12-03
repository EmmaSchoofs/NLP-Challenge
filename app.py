import streamlit as st
import random
import time
from crewai import Agent, Task, Crew, Process


task_values = []

def create_crewai_setup(query):
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Define Agents
    researcher = Agent(
        config=agents_config['content_ingestion_agent'],
        verbose=True
    )

    reporting_analyst = Agent(
        config=agents_config['question_answering_agent'],
        verbose=True
    )

    #Define Tasks
    research_task = Task(
        config=tasks_config['research_task'],
        agent = researcher
    )

    reporting_task = Task(
        config=tasks_config['reporting_task'].
        agent = reporting_analyst
    )

    crew = Crew(
        agents=[researcher, reporting_analyst],
        tasks=[research_task, reporting_task],
        verbose=2,
        process=Process.sequential,
    )

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

    query = st.text_input("ask me anything")

    if st.button("Go"):
        # Placeholder for stopwatch
            stopwatch_placeholder = st.empty()
            
            # Start the stopwatch
            start_time = time.time()
            with st.expander("Processing!"):
                sys.stdout = StreamToExpander(st)
                with st.spinner("Generating Results"):
                    crew_result = create_crewai_setup(product_name)

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