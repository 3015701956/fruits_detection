import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64

modelFile = 'ssd_mobilenet_frozen_inference_graph.pb'
configFile = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
classFile = 'coco_class_labels.txt'

# Create application title and file uploader widget.
st.title("Fruits Detection")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Read the TensorFlow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

with open(classFile) as fp:
    labels = fp.read().split('\n')

def detect_objects(net, img):
    """Run object detection over the input image."""
    # Blob dimension (dim x dim)
    dim = 300

    mean = (0, 0, 0)

    # Create blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (dim, dim), mean, True)

    # Pass blob to the network
    net.setInput(blob)

    # Perform Prediction
    objects = net.forward()
    return objects

def draw_text(im, text, x, y):
    """Draws text label at a given x-y position with a black background."""
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    # Get text size
    textSize = cv2.getTextSize(text, fontface, font_scale, thickness)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle.
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), fontface, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

def draw_objects(im, objects, threshold = 0.5):
    """Displays a box and text for each detected object exceeding the confidence threshold."""
    rows = im.shape[0]
    cols = im.shape[1]

    # For every detected object.
    for i in range(objects.shape[2]):
        # Find the class and confidence.
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Reover orginal coordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        # Check if the detection is of good quality
        if score > threshold:
            draw_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return im

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Load image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Or use PIL Image (which uses and RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

else:
    image = cv2.imread('fruit-vegetable.jpg')

# Create placeholders to display input and output images.
placeholders = st.columns(2)
# Display Input image in the first placeholder.
placeholders[0].image(image, channels='BGR')
placeholders[0].text("Input Image")

# Create a slider and get the threshold from the slider.
conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

# Call the fruits detection model to detect fruits in the image.
food_objects = detect_objects(net, image)

result = draw_objects(image.copy(), food_objects, conf_threshold)

# Display Detected fruits.
placeholders[1].image(result, channels='BGR')
placeholders[1].text("Output Image")

# Convert opencv image to PIL.
out_image = Image.fromarray(result[:, :, ::-1])
# Create a link for downloading the output file.
st.markdown(get_image_download_link(out_image, "fruits_output.jpg", 'Download Output Image'), unsafe_allow_html=True)
