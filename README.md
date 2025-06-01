# Exno.6-Prompt-Engg
# Register no.212222060221
# Aim: Development of Python Code Compatible with Multiple AI Tools

## DEVELOPMENT OF PYTHON CODE COMPATIBLE WITH MULTIPLEAI TOOLS
## Abstract
The rapid growth of Artificial Intelligence (AI) has led to the development of numerous
tools, libraries, and frameworks, each tailored for specific domains such as computer
vision, natural language processing, data science, and robotics. Python, being a dominant
language in the AI ecosystem, offers versatility and broad compatibility across various AI
platforms. This document explores the strategies, best practices, and design principles for
developing Python code that is compatible with multiple AI tools including TensorFlow,
PyTorch, scikit-learn, Hugging Face Transformers, and OpenAI's API, among others.
## Introduction
AI development often involves using multiple tools and libraries that may not natively
integrate with each other. Developing Python code that can interact with and leverage the
strengths of these tools enhances flexibility, reusability, and scalability. This document
outlines how to write modular, adaptable, and tool-agnostic Python code to facilitate
seamless integration across diverse AI platforms.
## Objective:

To understand how to send prompts to a large language model (LLM) API using Python
and interpret responses from the model.
Materials Required
- Python environment
- Internet Connection
- API Key for the LLM engine (Claude, OpenAI, etc.)
- API URL for the LLM endpoint
## Background
Large Language Models (LLMs) like Claude, GPT, etc., can generate human-like text
based on given prompts. These models are used in various applications, including chatbots,
language translation, content generation, and more.
In this experiment, we will use Python to communicate with an LLM by sending a prompt
and receiving a response. This exercise will demonstrate how to structure an API request,
send it the model, and display the result. import requests.
## Python Code
import requests
def query_llm_engine(prompt, api_url, api_key, model="claude", temperature=0.7):
"""
Query an LLM engine like Claude with a given prompt.
Parameters:
prompt (str): The prompt text to send to the engine.
api_url (str): The API endpoint URL for the LLM engine.
api_key (str): The API key for authenticating the request.
model (str): The model to use (default: "claude").
temperature (float): Sampling temperature (higher = more creative).
Returns:
str: The response text from the LLM engine, or an error message.
"""
headers = {
"Authorization": f"Bearer {api_key}",
"Content-Type": "application/json"
}
payload = {
"model": model,
"messages": [{"role": "user", "content": prompt}],
"temperature": temperature
}
## try:
response = requests.post(api_url, json=payload, headers=headers)
response.raise_for_status() # Raises an HTTPError if status is 4xx or 5xx
# Extract and return the response text
return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No
response received.")
except requests.exceptions.RequestException as e:
return f"Request error: {e}"
except ValueError:
return "Error parsing JSON response."
l
# Example usage:
if __name__ == "__main__":
api_url = "https://api.anthropic.com/v1/engines/claude/completions" # Replace with the
actual API URL
api_key = "YOUR_API_KEY" # Replace with your actual API key
prompt_text = "Explain the process of photosynthesis in simple terms."
response = query_llm_engine(prompt_text, api_url, api_key)
print("LLM Response:", response)
Example Observations
## After running the code, note down:
1. The prompt you used.
2. The response given by the model.
3. Any unexpected results or errors
## Observation
- Prompt: "Explain the process of photosynthesis in simple terms."
- Response: Photosynthesis is the process by which green plants, algae, and some
bacteriaconvert light energy from the sun into chemical energy stored in sugars like glucose.
Thisprocess is vital for life on Earth, as it produces oxygen and is the foundation of the
food chain.
During photosynthesis, plants take in carbon dioxide (CO₂) from the atmosphere and water
(H₂O) from the soil. These raw materials enter plant cells, particularly in the chloroplasts,
where the green pigment chlorophyll captures sunlight. Chlorophyll absorbs light energy,
which is then used to split water molecules, releasing oxygen (O₂) as a byproduct. The
energy from the sunlight excites electrons in the chlorophyll, and this energy is harnessed to
create two molecules, ATP (adenosine triphosphate) and NADPH, which act as energy
carriers.
The absorbed energy is then used to power a series of reactions that convert carbon dioxide
into glucose (C₆H₁₂O₆). This glucose provides an energy source for the plant and can be
stored for later use.
In essence, photosynthesis transforms sunlight into chemical energy, which sustains the
plant and, through the food chain, supports nearly all life on Earth.
## Analysis
Response Relevance: Did the response make sense for the given prompt :YES
Importance of Tool Compatibility
## Tool compatibility is crucial for the following reasons:
## Modularity: 
Allows components developed in one framework to be reused in another.
## Scalability: 
Facilitates transitioning from development to production.
## Interoperability:
Enhances the ability to use best-of-breed solutions from various
ecosystems.
## Efficiency:
Saves time and resources by avoiding redundant development.
## Key AI Tools and Frameworks
TensorFlow and Keras
TensorFlow is a comprehensive open-source platform for machine learning.
Keras, now integrated with TensorFlow, provides a user-friendly API for building
models.
## PyTorch
Known for its dynamic computation graph, PyTorch is popular in research and
production.
## Scikit-learn
A versatile library for traditional machine learning algorithms and data preprocessing.
## Hugging Face Transformers
A powerful library for state-of-the-art natural language processing.
## OpenAI API
Provides access to powerful language models for various NLP tasks.
Designing Compatible Python Code
Abstraction Layers
Create abstract base classes or interfaces that define common behaviors.

Implement tool-specific subclasses to handle variations.
## Configuration Management
Use configuration files (YAML, JSON) to define tool-specific parameters.
Employ environment variables to switch between toolkits.
## Dependency Management
Use requirements.txt or poetry for managing dependencies.
Employ virtual environments (venv, conda) to isolate tool setups.
## Common Data Structures
Use NumPy arrays or Pandas DataFrames as standard data formats.
Convert between formats as needed using utility functions.
## Best Practices
Modular Code Structure
Organize code into reusable modules and packages.
Logging and Debugging
Use the logging module to track operations across tools.
Error Handling
Implement robust error handling to manage tool-specific exceptions.
## Documentation
Document tool-specific requirements and usage in a README or docstring.
## Case Study: 
Multi-Tool NLP Pipeline
## Objective:
Build a text classification pipeline using both Hugging Face Transformers and
## scikit-learn.
## Steps:
Use Hugging Face for tokenization and feature extraction.
Convert features to NumPy arrays.
Use scikit-learn for classification (e.g., Logistic Regression).
Evaluate the model and log results.
## Outcome: 
A modular pipeline with interchangeable components.
Challenges and Solutions
Version Conflicts
Use Docker or virtual environments to manage dependencies.
API Incompatibilities
Abstract APIs to decouple code from specific tool implementations.
Performance Overheads
Profile code using cProfile or line_profiler.
Future Trends
## Unified Frameworks:
Libraries like ONNX are enabling model interchangeability.
## AutoML:
Tools that abstract the complexity of using multiple libraries.
Standardized APIs: Initiatives aiming to unify AI tool APIs.
## Conclusion
Developing Python code that is compatible with multiple AI tools is essential for building
robust, flexible, and scalable AI systems. Through thoughtful design, modular architecture,
and adherence to best practices, developers can harness the strengths of various AI
libraries, enhancing the overall efficiency and capability of their applications.

# Result: The corresponding Prompt is executed successfully
