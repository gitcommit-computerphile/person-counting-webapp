import streamlit as st
import requests
from PIL import Image, ImageDraw

API_URL = "http://person-counter-api:5000/count-persons"


st.set_page_config(page_title="Person Counter", page_icon="🧍", layout="wide")

st.title("🧍 Person Counter")
st.write("Upload an image and detect people with YOLO.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        conf = det["confidence"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 15)), f"Person {conf}", fill="red")

    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    if st.button("Count persons"):
        with st.spinner("Sending image to backend..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            try:
                response = requests.post(API_URL, files=files, timeout=120)

                if response.status_code == 200:
                    data = response.json()
                    boxed = draw_boxes(image, data["detections"])

                    with col2:
                        st.subheader("Detected")
                        st.image(boxed, use_container_width=True)

                    st.success(f'Persons detected: {data["person_count"]}')
                    st.json(data)

                else:
                    st.error(f"Backend error: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to backend: {e}")