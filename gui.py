import streamlit as st
from PIL import Image
import cv2 as cv
from displayTumor import DisplayTumor
from predictTumor import predictTumor

# Initialize DisplayTumor class
DT = DisplayTumor()

# Set page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Title and description
st.title("Brain Tumor Detection System")
st.write("Upload an MRI image to detect or view tumor regions.")

# Sidebar for user options
action = st.sidebar.radio(
    "Select an Action:",
    ["Detect Tumor", "View Tumor Region"],
    help="Choose what you want to do with the uploaded MRI image."
)

# File uploader for image input
uploaded_file = st.file_uploader("Upload MRI Image (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file:
    # Read the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Convert the image to OpenCV format
    mri_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Check user action
    if st.button("Process"):
        if action == "Detect Tumor":
            # Call tumor detection function
            result = predictTumor(mri_image)

            # Display result
            if result > 0.5:
                st.error("**Tumor Detected** in the MRI Image.", icon="🚨")
            else:
                st.success("**No Tumor Detected** in the MRI Image.", icon="✅")

        elif action == "View Tumor Region":
            # Call DisplayTumor function to process and display tumor region
            DT.readImage(image)
            tumor_image = DT.displayTumor()  # Assuming this returns a processed image

            if tumor_image is not None:
                # Convert OpenCV image to PIL format for display
                tumor_image = Image.fromarray(cv.cvtColor(tumor_image, cv.COLOR_BGR2RGB))
                st.image(tumor_image, caption="Tumor Region Highlighted", use_column_width=True)
            else:
                st.warning("Unable to highlight tumor region. Please ensure a valid MRI image is uploaded.")

else:
    st.info("Please upload an MRI image to proceed.")
