import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np
from displayTumor import DisplayTumor
from predictTumor import predictTumor

# Initialize the DisplayTumor class
DT = DisplayTumor()

# Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# App title and description
st.title("Brain Tumor Detection System")
st.write("Upload an MRI image to detect tumors or highlight tumor regions.")

# Sidebar options for user actions
action = st.sidebar.radio(
    "Select an Action:",
    ["Detect Tumor", "View Tumor Region"],
    help="Choose what you want to do with the uploaded MRI image."
)

# File uploader for MRI image
uploaded_file = st.file_uploader("Upload MRI Image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

# Main functionality to process the uploaded file
if uploaded_file:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Convert the image to OpenCV format
    mri_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Process based on the user action
    if st.button("Process"):
        if action == "Detect Tumor":
            try:
                # Call tumor detection function
                result = predictTumor(mri_image)

                # Display detection results
                if result > 0.5:
                    st.error("**Tumor Detected** in the MRI Image.", icon="🚨")
                else:
                    st.success("**No Tumor Detected** in the MRI Image.", icon="✅")
            except Exception as e:
                st.error(f"An error occurred during tumor detection: {str(e)}")

        elif action == "View Tumor Region":
            try:
                # Process and highlight the tumor region
                DT.readImage(image)
                tumor_image = DT.displayTumor()

                if tumor_image is not None:
                    # Convert OpenCV image to PIL format for Streamlit display
                    tumor_image = Image.fromarray(cv.cvtColor(tumor_image, cv.COLOR_BGR2RGB))
                    st.image(tumor_image, caption="Highlighted Tumor Region", use_column_width=True)
                else:
                    st.warning("No tumor region could be highlighted. Please ensure the uploaded image is valid.")
            except Exception as e:
                st.error(f"An error occurred while highlighting the tumor region: {str(e)}")
else:
    st.info("Please upload an MRI image to proceed.")
