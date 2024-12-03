from flask import Flask, request, jsonify
from personalized_learning_assistant.src.personalized_learning_assistant.crew import PersonalizedLearningAssistant

app = Flask(__name__)

# Initialize CrewAI
crew = PersonalizedLearningAssistant().crew()

@app.route('/process', methods=['POST'])
def process_input():
    """
    Endpoint to process input using CrewAI.
    """
    print("Received JSON:", request.json)
    try:
        # Log the incoming request
        app.logger.debug(f"Received request: {request.json}")

        input_text = request.json.get("input", None)  # Get JSON data from request
        if not input_text:
            return jsonify({'error': 'Invalid input format'}), 400

        # Prepare input for CrewAI if necessary
        crew_input = {"topic": input_text}  # Adjust key based on crew.kickoff requirements

        # Use CrewAI to process the input
        result = crew.kickoff(inputs=crew_input)

        return jsonify({'result': result}), 200
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
