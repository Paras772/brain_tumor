import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
from displayTumor import displayTumor
from predictTumor import predictTumor

# Initialize the DisplayTumor class
DT = displayTumor()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="centered",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Brain Tumor Detection System")
st.write("This application allows you to upload an MRI image to detect or visualize tumor regions.")

# Sidebar for selecting user action
st.sidebar.header("Choose Action")
action = st.sidebar.radio(
    "Select an Action:",
    options=["Detect Tumor", "View Tumor Region"],
    help="Choose between detecting a tumor or highlighting the tumor region."
)

# Upload MRI image
uploaded_file = st.file_uploader(
    "Upload MRI Image (jpg, jpeg, png):",
    type=["jpg", "jpeg", "png"],
    help="Supported formats are JPG, JPEG, and PNG."
)

# Process the uploaded file
if uploaded_file:
    try:
        # Load the image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Convert the image to OpenCV format
        mri_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

        # Action selection
        if st.button("Process"):
            if action == "Detect Tumor":
                try:
                    # Perform tumor detection
                    result = predictTumor(mri_image)

                    # Display the result
                    if result > 0.5:
                        st.error("**Tumor Detected** in the MRI Image.", icon="🚨")
                    else:
                        st.success("**No Tumor Detected** in the MRI Image.", icon="✅")
                except Exception as e:
                    st.error(f"An error occurred during tumor detection: {str(e)}")

            elif action == "View Tumor Region":
                try:
                    # Highlight the tumor region
                    DT.readImage(image)
                    tumor_image = DT.displayTumor()

                    if tumor_image is not None:
                        # Convert the result to PIL format for display
                        tumor_image = Image.fromarray(cv.cvtColor(tumor_image, cv.COLOR_BGR2RGB))
                        st.image(tumor_image, caption="Tumor Region Highlighted", use_column_width=True)
                    else:
                        st.warning("No tumor region could be highlighted. Ensure the uploaded image is valid.")
                except Exception as e:
                    st.error(f"An error occurred while highlighting the tumor region: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred while processing the uploaded image: {str(e)}")
else:
    st.info("Please upload an MRI image to proceed.")

# Footer
st.sidebar.markdown("### About")
st.sidebar.info("This application uses deep learning to analyze MRI images and detect brain tumors.")
