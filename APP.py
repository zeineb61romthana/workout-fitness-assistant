import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import DocArrayInMemorySearch

MODEL = "llama2"
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings()


model_params = {
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 0.7,
    "repetition_penalty": 1.0
}

def generate_text(prompt, params):
    try:
        response = model.invoke(prompt, **params)
        return response
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return "Error generating response."

@st.cache_data
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        return loader.load_and_split()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

def get_answer(question, context):
    try:
        chain = prompt | model | parser
        response = chain.invoke({"context": context, "question": question})
        return response
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        return "Error getting answer."


def process_in_batches(pages, batch_size=500):
    text_batches = []
    current_batch = []
    current_size = 0

    for page in pages:
        page_text = page.content
        if current_size + len(page_text) > batch_size:
            text_batches.append(" ".join(current_batch))
            current_batch = []
            current_size = 0
        current_batch.append(page_text)
        current_size += len(page_text)

    if current_batch:
        text_batches.append(" ".join(current_batch))

    return text_batches

motivational_messages = [
    "Keep pushing forward! Every workout counts!",
    "You're doing great! Stay focused and motivated!",
    "Remember, progress takes time. Keep up the hard work!",
    "Believe in yourself and all that you are capable of!",
    "Success is a journey, not a destination. Keep going!"
]

def get_random_motivational_message():
    import random
    return random.choice(motivational_messages)

def main():
    st.title("Workout & Fitness Assistant")

    
    file_path = "C:\\Users\\me\\OneDrive\\Bureau\\SPORTIFY_AI\\exemple_1_\\workout pdf paper.pdf"  

    
    st.write("Welcome to your personal Workout & Fitness Assistant!")
    st.write("I'm here to help you with workout tips, fitness advice, and answer your questions related to fitness.")

    
    if os.path.isfile(file_path):
        
        pages = load_pdf(file_path)

        if pages:
            
            try:
                vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
                retriever = vectorstore.as_retriever()

                st.write("Enter your question related to workout and fitness:")
                question = st.text_input("Type your question here:")

                if st.button("Get Answer") and question:
                    
                    batches = process_in_batches(pages)
                    answers = []

                    for batch in batches:
                        context = batch
                        answer = get_answer(question, context)
                        answers.append(answer)

                    final_answer = " ".join(answers)
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {final_answer}")

                    
                    st.write(f"**Motivational Message:** {get_random_motivational_message()}")

            except Exception as e:
                st.error(f"Error creating vector store: {e}")
        else:
            st.error("The PDF document could not be loaded. Please check the file and try again.")
    else:
        st.error("The file path provided does not exist. Please check the path and try again.")

if __name__ == "__main__":
    main()
