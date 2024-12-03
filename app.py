import streamlit as st
import requests
import time

# Flask API URL
FLASK_API_URL = "http://127.0.0.1:5000/process"  # Change this if needed

# App Header
st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Make a request to the Flask API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Send user input to Flask API
            response = requests.post(
                FLASK_API_URL,
                json={"input": prompt},  # Adjust key if Flask expects a different field
                timeout=5  # Optional: Add timeout to handle delays
            )

            # Check if the API call was successful
            if response.status_code == 200:
                full_response = response.json().get("result", "No response from the assistant.")
            else:
                full_response = f"Error: Received status code {response.status_code} from Flask API."

        except requests.exceptions.RequestException as e:
            full_response = f"Error: Unable to reach Flask API. Details: {str(e)}"

        # Simulate typing effect
        for chunk in full_response.split():
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
