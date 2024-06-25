import streamlit as st
import fitz  # PyMuPDF
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as document:
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    return text

# Load the tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

# Initialize Streamlit app
st.title("Policy Document Question Answering")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    # Extract text from uploaded PDF
    document_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text for verification (optional)
    st.write("Extracted Text from PDF:")
    st.write(document_text)

    # Load the QA pipeline
    qa_pipeline = load_model()

    # Question input
    question = st.text_input("Ask a question about the policy document:")

    if question:
        # Generate response
        answer = qa_pipeline(question=question, context=document_text)['answer']
        st.write(f"**Answer:** {answer}")
