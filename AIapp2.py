import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import FAISS
#from PyPDF2 import PdfReader
#from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load environment variables from folder.env
load_dotenv()

# Initialize Google API from the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")

# Initialize the models with different temperature settings
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize Google Generative AI embeddings (mocked here)
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vector_db = None  # Will be initialized when a document is processed

# # Function to load and read PDF
# def read_pdf(file):
#     pdf_reader = PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Chunking and FAISS storage
# def process_and_store_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     embeddings = embedding_model.embed_documents(chunks)  # Using Google Generative AI for embeddings
#     global vector_db
#     vector_db = FAISS.from_embeddings(embeddings, chunks)
#     return chunks

# # Function to search the FAISS database with embeddings
# def query_faiss_db(query):
#     query_embedding = embedding_model.embed_query(query)
#     docs = vector_db.similarity_search(query_embedding)
#     return docs

# # Function for Research AI
# def research_task(query):
#     research_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     messages = [
#         (
#             "system",
#             "You are a Research Assistant. Please research the following topic: {query} and provide a comprehensive overview.",
#         ),
#         (
#             "human", query
#         ),
#     ]
  
#     research_response = research_model.invoke(messages)
#     return research_response.content

# # Function for Summarizer AI
# def summarizer_task(text):
#     research_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     message = [
#         (
#             "system",
#             "Summarize the following text in a concise manner:{text}",

#         ),
#         (
#             "human", text
#         )
#     ]
#     summary_response = research_model.invoke(message)
#     return summary_response.content

# # Function for Elaborate AI
# def elaborate_task(text):
#     elaborate_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     message = [
#         (
#             "system",
#             "Elaborate on the following text and provide more detailed information: {text}",

#         ),
#         (
#             "human", text
#         )
#     ]
#     elaborate_response = elaborate_model.invoke(message)
#     return elaborate_response.content

# # Function for Elaborate AI
# def pdf_chat_task(text,question):
#     chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     message = [
#         {
#             "role": "system", "content":

#             "You are a helpfull assistant that answers question based on the provided text."},

#         {

#             "role" : "user", "content":

#             f"Here is the PDF content:{text}"},

#         {
#             "role" : "user", "content": question}
    
        
#         ]
#     elaborate_response = chat_model.invoke(message)
#     return elaborate_response.content

# Function for Chatbot (powered by Google Generative AI)
def chatbot_task(user_query):
    chatbot_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chatbot_response = chatbot_model.invoke(user_query)
    return chatbot_response.content

# Function for Voice Chatbot (powered by Google Generative AI)
def voice_chatbot_task(user_query):
    voice_chatbot_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    voice_chatbot_response = voice_chatbot_model.invoke(user_query)
    return voice_chatbot_response.content

def evaluate_code(question, code: str):
    # Modify the system message for evaluation
    code_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    message = [
        ("system", f"Evaluate the following code based on question {question} and provide the Correction = %, Performance = % do not include another information: {code} if any error occurs in the code please find the error and highlight it. do not correct the code and display the correction/performance percentage"),
        ("human", code)
    ]
    
    # Send the message to the model for evaluation
    code_response = code_model.invoke(message)
    return code_response.content

def evaluate_answer(question, answer):
    # Modify the system message for evaluation
    code_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    message = [
        ("system", f"Check/evaluate the following answer based on question {question} and provide the Pass = % and Fail = %,  if the percentage is more than 60% then declare his pass ans show only pass % or if the percentage is less than 60% declare him as Fail ans show only fail %. Do not include another information: {answer}"),
        ("human", answer)
    ]
    
    # Send the message to the model for evaluation
    code_response = code_model.invoke(message)
    return code_response.content


def evaluate_voice_answer(question,answer):
    # Modify the system message for evaluation
    code_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    message = [
        ("system", f"You are a helpfull assistant that check the answer based on question {question} and evaluate the following answer is correct or not and provide the Pass = % and Fail = %,  if the percentage is more than 60% then declare his pass ans show only pass % or if the percentage is less than 60% declare him as Fail ans show only fail %. Do not include another information: {answer}"),
        ("human", answer)
    ]
    
    # Send the message to the model for evaluation
    code_response = code_model.invoke(message)
    return code_response.content


# Streamlit UI
st.title("AI-Powered Streamlit App")

# # Sidebar for task selection
# task = st.sidebar.selectbox(
#     "Select a Task", 
#     ["Research AI", "Summarizer AI", "Elaborate AI", "Chat PDF", "Chatbot", "Voice Chat",
#     "MCQ Question", "Code Evaluation", "Answer Evaluation", "Voice Answer Evaluation"]
#     )

task = st.sidebar.selectbox(
    "Select a Task", 
     ["Chatbot", "Voice Chat", "MCQ Question", "Code Evaluation", "Answer Evaluation", "Voice Answer Evaluation"]
     )

# # Research AI Task
# if task == "Research AI":
#     st.title("Welcome to Research AI")
#     research_query = st.text_input("Enter the research topic:")
#     if st.button("Research"):
#         research_result = research_task(research_query)
#         st.write(research_result)

# # Summarizer AI Task
# elif task == "Summarizer AI":
#     st.title("Welcome to Summarizer App")
#     summary_query = st.text_area("Enter the text to summarize:")
#     if st.button("Summarize"):
#         summary_result = summarizer_task(summary_query)
#         st.write(summary_result)

# # Elaborate AI Task
# elif task == "Elaborate AI":
#     st.title("Elaborate the Topic or Content")
#     elaborate_query = st.text_area("Enter the text to elaborate:")
#     if st.button("Elaborate"):
#         elaborate_result = elaborate_task(elaborate_query)
#         st.write(elaborate_result)

# elif task == "Chat PDF":
#     st.title("Chat with Uploaded PDF")
#     uploaded_file = st.file_uploader("Uploade PDF",type = ['pdf'])
#     text = read_pdf(uploaded_file)
#     question = st.text_input("Ask from Pdf:")
#     if st.button("ask"):
#         result = pdf_chat_task(text=text,question=question)
#         st.write(result)

# Chatbot Task
if task == "Chatbot":
    st.title("Welcome to Chatbot")
    user_query = st.text_input("Ask anything from ChatBot:")
    if st.button("Ask"):
        result = chatbot_task(user_query)
        st.write(result)

#Voice Chatbot Task
elif task == "Voice Chat":
    st.title("Welcome to Voice Chatbot")
    
    def speak_custom_voice(response_text):
        # Example of using your custom voice model
        # For a local model, you would use something like Coqui or a local API call.
        # If using a service like Descript, you can send API requests to generate audio.
        # This is a placeholder: replace with actual function to generate custom speech
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Adjust speech speed
        engine.setProperty('volume', 1)  # Set volume level
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)  # Set the voice
        engine.say(response_text)
        engine.runAndWait()
    
    if st.button("Speak your Query"):
        # Using the microphone as the source for input
        # Setup the speech recognition part
        with sr.Microphone() as source:
            # Initialize speech recognizer and TTS engine
            # Initialize the recognizer and TTS engine
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            st.write("Listening...")
            audio = recognizer.listen(source)
        
        try:
            user_query = recognizer.recognize_google(audio)
            st.write(f"You said: {user_query}")
            
            # Get chatbot response
            result = voice_chatbot_task(user_query)
            st.write(f"Answer: {result}")

            # Speak the response using the custom voice model
            speak_custom_voice(result)
            

        except Exception as e:
            print(f"Error: {e}")


elif task == "MCQ Question":
    st.title("MCQ Test for Students")
    # MCQ Questions and Answers
    questions = [
        {"question": "What is the capital of France?", 
        "options": ["Berlin", "Madrid", "Paris", "Rome"], 
        "correct": "Paris"},
        
        {"question": "Which language is primarily used for data science?", 
        "options": ["Java", "Python", "C++", "JavaScript"], 
        "correct": "Python"},
        
        {"question": "What is the largest ocean on Earth?", 
        "options": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"], 
        "correct": "Pacific Ocean"},
        
        {"question": "Who developed the theory of relativity?", 
        "options": ["Isaac Newton", "Albert Einstein", "Nikola Tesla", "Galileo Galilei"], 
        "correct": "Albert Einstein"}
    ]
    # Create a function to calculate the score
    def calculate_score(student_answers):
        correct_answers = 0
        for i, answer in enumerate(student_answers):
            if answer == questions[i]['correct']:
                correct_answers += 1
        return correct_answers
    # Collect student answers
    student_answers = []
    for q in questions:
        st.subheader(q['question'])
        
        # Use index=None to make sure no option is pre-selected
        answer = st.radio(
            "Choose one option",
            q['options'],
            index=None,  # This ensures no option is selected by default
            key=q['question']  # Unique key for each question
        )
        student_answers.append(answer)
    # Submit Button
    if st.button("Submit Test"):
        # Calculate score
        score = calculate_score(student_answers)
        total_questions = len(questions)
        
        # Display score
        st.write(f"Your score is: {score}/{total_questions}")
        
        # Display the correct answers
        st.write("Correct answers:")
        for i, q in enumerate(questions):
            if student_answers[i] != q['correct']:
                st.write(f"Question {i + 1}: The correct answer is '{q['correct']}'")

        # Create a pie chart of correct/incorrect answers
        correct = sum([1 for i in range(total_questions) if student_answers[i] == questions[i]['correct']])
        incorrect = total_questions - correct

        # Pie chart visualization
        fig, ax = plt.subplots()
        ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
        st.pyplot(fig)

elif task == "Code Evaluation":
    st.title("Python Code Evaluation App")
    # List of sample Python questions
    questions = [
        "Write a function to calculate the factorial of a number.",
        "Write a Python function that returns the Fibonacci sequence up to a given number.",
        "Create a Python function that checks whether a string is a palindrome.",
        "Write a Python program to sort a list of integers in ascending order."
    ]
        
    # Display a sample question to the user
    st.subheader("Sample Python Question:")
    question = st.selectbox("Choose a question", questions)
    st.write(f"**Question:** {question}")

    # Step 2: Get the student's code input
    st.subheader("Your Code:")
    student_code = st.text_area("Write your Python code below:")
    
    if st.button("Evaluate Code"):
        if student_code:
            # Step 3: Evaluate the code
            st.subheader("Evaluation from the Model:")
            evaluation = evaluate_code(question,student_code)
            st.write(evaluation)
        else:
            st.warning("Please write your code before submitting.")

elif task == "Answer Evaluation":
    st.title("subjective question Evaluation App")
    # List of sample Python questions
    questions = [
        "What is automation testing?",
        "What is oops in python programming language?",
        "what is qa testing?",
    ]
    
    # Display a sample question to the user
    st.subheader("Sample Question:")
    question = st.selectbox("Choose a question", questions)
    st.write(f"**Question:** {question}")

    # Step 2: Get the student's code input
    st.subheader("Your Answer:")
    answer = st.text_area("Write your answer below:")
    if st.button("Check Answer"):
        if answer:
            # Step 3: Evaluate the code
            st.subheader("Evaluation from the Model:")
            evaluation = evaluate_answer(question,answer)
            st.write(evaluation)
        else:
            st.warning("Please write your code before submitting.")

elif task == "Voice Answer Evaluation":
    st.title("voice question answer Evaluation App")
    # List of sample Python questions
    questions = [
        "what is capital of india?",
        "where is taj mahal?",
        "who is pm of india?",
    ]
    
    # Display a sample question to the user
    st.subheader("Sample Questions:")
    question = st.selectbox("Choose a question", questions)
    st.write(f"**Question:** {question}")

    # Step 2: Get the student's  input
    st.subheader("Your Answer in 200 words:")
    if st.button("Speak your answer"):
        # Using the microphone as the source for input
        # Setup the speech recognition part
        with sr.Microphone() as source:
            # Initialize speech recognizer and TTS engine
            # Initialize the recognizer and TTS engine
            recognizer = sr.Recognizer()
            recognizer.adjust_for_ambient_noise(source)
            st.write("Listening...")
            audio = recognizer.listen(source)
            answer = recognizer.recognize_google(audio)
            st.write(f"You said: {answer}")
            
            # Step 3: Evaluate the answer
            st.subheader("Evaluation from the Model:")
            evaluation = evaluate_voice_answer(question,answer)
            st.write(evaluation)