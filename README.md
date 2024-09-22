# 🏋️‍♂️ Workout & Fitness Assistant AI 🌟

This project is a **Workout & Fitness Assistant** built using advanced AI models to provide users with personalized fitness advice, workout tips, and motivational messages. The application processes fitness-related PDFs, extracts content, and answers users' workout questions in real-time.

### ✨ Key Features:

- **PDF Parsing**: Extracts fitness-related content from PDFs using `PyPDFLoader`.
- **Question Answering**: Provides accurate answers to user-submitted questions using a custom AI pipeline powered by the **Llama2** model.
- **Batch Processing**: Efficiently processes large documents in manageable chunks to provide context-aware responses.
- **Motivational Messages**: Delivers a random motivational message with each response to keep you inspired and focused on your fitness journey!

### 🚀 Libraries and Tools:

- **Selenium**: Automates browser interactions for web scraping.
- **BeautifulSoup**: Parses HTML for data extraction.
- **Streamlit**: Provides a clean and interactive UI.
- **LangChain**: Handles prompt templates and model invocations.
- **DocArrayInMemorySearch**: Creates vector stores for efficient document retrieval and querying.

### 🛠️ Technologies:

- `Python`
- `Streamlit`
- `LangChain`
- `PyPDFLoader`
- `Ollama` (Llama2 for natural language processing)

### 💻 How It Works:

1. Load a fitness-related PDF file.
2. Submit your workout or fitness question in the app.
3. The assistant processes the PDF content and provides an AI-powered response.
4. Enjoy motivational messages to stay on track with your fitness goals!
