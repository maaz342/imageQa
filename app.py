import streamlit as st
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image, ExifTags
import torch
import os
import numpy as np

# Custom CSS for styling
st.set_page_config(page_title="Image Insight: Captioning and QA", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        color: #0066cc;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #333333;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
    }
    .file-upload {
        text-align: center;
        margin-top: 20px;
    }
    .generate-caption-btn, .answer-btn {
        background-color: #0066cc;
        color: #ffffff;
        font-size: 18px;
        border-radius: 10px;
        margin-top: 20px;
        width: 100%;
        padding: 10px;
    }
    .question-input {
        margin-top: 20px;
        width: 100%;
    }
    .answer {
        color: #333333;
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_caption_model():
    try:
        model_name = "Salesforce/blip-image-captioning-base"
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        processor = BlipProcessor.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        st.error(f"Error loading caption model: {e}")
        return None, None

def get_image_caption(model, processor, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def answer_question(image, uploaded_file, question):
    question = question.lower()

    if "dimension" in question or "size" in question:
        width, height = image.size
        return f"The image dimensions are {width}x{height} pixels."
    elif "file size" in question or "size" in question:
        file_size = os.path.getsize(uploaded_file)
        return f"The image file size is {file_size} bytes."
    elif "mode" in question:
        mode = image.mode
        return f"The image mode is {mode}."
    elif "format" in question:
        format = image.format
        return f"The image format is {format}."
    elif "aspect ratio" in question:
        width, height = image.size
        aspect_ratio = width / height
        return f"The aspect ratio of the image is {aspect_ratio:.2f}."
    elif "average color" in question or "average colour" in question:
        np_image = np.array(image)
        avg_color = np.mean(np_image, axis=(0, 1))
        avg_color = [int(x) for x in avg_color]
        return f"The average color of the image is RGB{tuple(avg_color)}."
    elif "brightness" in question:
        np_image = np.array(image.convert('L'))
        avg_brightness = np.mean(np_image)
        return f"The average brightness of the image is {avg_brightness:.2f}."
    elif "exif" in question or "metadata" in question:
        exif_data = image._getexif()
        if exif_data is not None:
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            exif_info = "\n".join([f"{k}: {v}" for k, v in exif.items()])
            return f"EXIF metadata:\n{exif_info}"
        else:
            return "No EXIF metadata found."
    else:
        return "I can answer questions about the image dimensions, file size, mode, format, aspect ratio, average color, brightness, and EXIF metadata."

def main():
    st.markdown("<h1 class='title'>🖼️ Image Insight: Captioning and QA</h1>", unsafe_allow_html=True)

    model, processor = load_caption_model()
    if model is None:
        st.stop()

    st.markdown("<div class='file-upload'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        st.markdown("<h2 class='subheader'>Generate Caption</h2>", unsafe_allow_html=True)
        if st.button("Generate Caption", key="caption_btn", help="Generate a caption for the uploaded image"):
            with st.spinner('Generating caption...'):
                caption = get_image_caption(model, processor, image)
                st.markdown(f"<div class='answer'><b>Caption:</b> {caption}</div>", unsafe_allow_html=True)
                st.session_state.caption = caption

        st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
        if "caption" in st.session_state:
            question = st.text_input("Ask a question about the image", key="question_input", placeholder="Type your question here...")
            if st.button("Get Answer", key="answer_btn", help="Get an answer to your question about the image"):
                if question:
                    with st.spinner('Getting answer...'):
                        answer = answer_question(image, uploaded_file, question)
                        st.markdown(f"<div class='answer'><b>Answer:</b> {answer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
