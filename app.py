from PIL import Image
import streamlit as st
import os
import pandas as pd
import base64
import google.generativeai as gem
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import ast

# Set page configuration
st.set_page_config(
    page_title="Business Card Extractor",
    page_icon="📇",
    layout="wide",
)

# Custom CSS for background color and UI adjustments
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E0F7FA; /* Light blue background */
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5; /* Blue color */
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #B3E5FC; /* Light blue sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add logo to the sidebar
logo_path = "logo.png"  # Replace with your logo file path
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)

# Title
st.markdown('<h1 style="text-align: center; color: #1E88E5;">Business Card Extractor</h1>', unsafe_allow_html=True)

# Configuration
IMAGE_FOLDER = "uploaded_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize Google Generative AI
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDc8qqXA7dKKExF5sm_dFISijUU5vHatls'  # Add your Google API key here
gem.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process images and extract data
def process_images(selected_images):
    columns = ["Person name", "Company name", "Email", "Contact number"]
    all_rows = []
    json_data = {}

    for image_file in selected_images:
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        image_base64 = image_to_base64(image_path)
        vision = gem.GenerativeModel('gemini-1.5-flash-latest')
        res = vision.generate_content([
            """You are only a business card image recognizer, you will tell clean 'YES' if it is it else clean 'NO' """,
            {"mime_type": "image/jpeg", "data": image_base64}
        ])
        if res.text == 'NO':
            st.info(f"{os.path.basename(image_path)} is not a business card", icon='❗')
            continue

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Carefully analyze the business card(s) and get the output in pure json format

                    [{"Person name": "full name of the person if exists",
                        "Company name": "get the full company name if exists",
                        "Email": "get the complete mail if exists",
                        "Contact number": "get every contact number if exists"}]

                    if a card has multiple person name then the output be like:

                    [{"Person name": "full name of the person if exists",
                        "Person name 2": "full name of the person if exists",
                        "Company name": "get the full company name if exists",
                        "Email": "get the complete mail if exists",
                        "Contact number": "get every contact number if exists"}]
                        your response shall not contain ' ```json ' and ' ``` ' """,
                },
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        )

        try:
            response = llm.invoke([message])
            response = response.content.replace('null', 'None')
            extracted_data = ast.literal_eval(response)

            for item in extracted_data:
                person_name = item.get("Person name", "")
                person_name_2 = item.get("Person name 2", "")
                row = {
                    "Person name": f"{person_name}, {person_name_2}",
                    "Company name": item.get("Company name", ""),
                    "Email": item.get("Email", ""),
                    "Contact number": item.get("Contact number", ""),
                }
                all_rows.append(row)

            json_data[image_file] = extracted_data
        except Exception as e:
            st.error(f"Failed to process image: {image_file}")
            st.exception(e)

    return all_rows, json_data

# File uploader
uploaded_files = st.file_uploader("Upload Files (PDF, DOCX, Images)", accept_multiple_files=True, type=["pdf", "docx", "jpg", "jpeg", "png"])

# Display uploaded images
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(IMAGE_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!", icon="✅")

    # Display images in a grid
    image_paths = [os.path.join(IMAGE_FOLDER, uploaded_file.name) for uploaded_file in uploaded_files if uploaded_file.name.lower().endswith(('jpg', 'jpeg', 'png'))]
    if image_paths:
        num_cols = 3  # Number of columns for grid layout
        cols = st.columns(num_cols)
        for i, image_path in enumerate(image_paths):
            with cols[i % num_cols]:
                image = Image.open(image_path)
                st.image(image, caption=os.path.basename(image_path))

# Process selected images
if st.button("Extract Data"):
    selected_images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not selected_images:
        st.error("No images found to process.")
    else:
        all_rows, json_data = process_images(selected_images)

        # Display extracted data in a DataFrame
        if all_rows:
            df = pd.DataFrame(all_rows, columns=["Person name", "Company name", "Email", "Contact number"])
            st.markdown('##### Extracted Data')
            edited_df = st.data_editor(df, num_rows="dynamic", key="editor_displayed")

            # Download CSV
            csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="business_cards.csv",
                mime="text/csv",
            )

# Clear all data
if st.button("Clear All Data"):
    for f in os.listdir(IMAGE_FOLDER):
        os.remove(os.path.join(IMAGE_FOLDER, f))
    st.info("All data cleared!", icon="❗")
