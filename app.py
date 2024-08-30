import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
import time
import json

# Load the text data
file_path = '/content/radiologue.csv'
text_data = pd.read_csv(file_path)

# Initialize the OpenAI model
llm = OpenAI(api_key='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')

# Define the prompt template for health statistics data
prompt_template = PromptTemplate(
    input_variables=["data_description", "question"],
    template="""
    You are a radiology doctor in the Caribbean. You have access to the following health statistics data:
    {data_description}

    Question: {question}

    Please provide a detailed answer based on the data.
    """
)

# Create the LangChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate data description for the health statistics
def generate_data_description():
    sample_entries = text_data.sample(min(len(text_data), 5))  # Get up to 5 random rows
    description = "The dataset contains various health statistics. It includes data points such as Infant Mortality Rate, Life Expectancy, Maternal Mortality Rate, Prevalence of Diabetes, and Prevalence of Hypertension. The data is structured with the following columns:\n"
    description += "- Keyword: A list of radiology terms.\n"
    description += "- Response: Definitions or meanings of all the terms.\n\n"
    description += "Example entries:\n"
    description += "\n".join(f"Keyword: {row['Keyword']}, Response: {row['Response']}" for _, row in sample_entries.iterrows())
    return description

def get_response(question):
    data_description = generate_data_description()
    st.write("Data Description:", data_description)  # Debugging line to check data description
    attempt = 0
    while True:
        try:
            response = chain.run(data_description=data_description, question=question)
            st.write("Model Response:", response)  # Debugging line to check model response
            return response
        except Exception as e:
            error_message = str(e)
            if 'Rate limit' in error_message or 'quota' in error_message:
                wait_time = 2 ** attempt  # Exponential backoff
                st.write(f"Rate limit exceeded: {error_message}. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                attempt += 1
                if attempt > 5:  # Limit the number of retries
                    st.write("Exceeded maximum retry attempts.")
                    break
            else:
                st.write(f"An error occurred: {e}")
                break

# Streamlit App
st.title("Radiology Consultation and Appointment Scheduling")

# Display the question and answer section
st.header("Ask a Question")
user_question = st.text_input("Enter your question here:")

if st.button("Submit Question"):
    if user_question:
        answer = get_response(user_question)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")

# Display the appointment scheduling section
st.header("Schedule an Appointment")
with st.form(key='appointment_form'):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    date = st.date_input("Preferred Date")
    time_slot = st.time_input("Preferred Time")
    submit_button = st.form_submit_button("Book Appointment")
    
    if submit_button:
        if name and email and phone and date and time_slot:
            appointment_data = {
                "name": name,
                "email": email,
                "phone": phone,
                "date": date.strftime('%Y-%m-%d'),
                "time": time_slot.strftime('%H:%M:%S')
            }
            
            # Save appointment data to a file (or database)
            try:
                with open('appointments.json', 'a') as file:
                    file.write(json.dumps(appointment_data) + "\n")
                st.success("Appointment booked successfully!")
            except Exception as e:
                st.error(f"An error occurred while booking the appointment: {e}")
        else:
            st.warning("Please fill in all fields.")
