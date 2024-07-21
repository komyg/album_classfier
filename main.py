import os
from dotenv import load_dotenv
from PIL import Image
from PIL.ImageFile import ImageFile
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2
import numpy as np

load_dotenv()

detect_client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"]
)
infer_client = InferenceHTTPClient(
    api_url="https://infer.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"]
)


def show_annotations(detections, inference_image):
    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=inference_image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    # display the image
    sv.plot_image(annotated_image)


def locate_empty_cards(image: ImageFile) -> sv.Detections:
    empty_cards = detect_client.infer(inference_input=image, model_id="album-wrkp7/1")
    detections = sv.Detections.from_inference(empty_cards)
    return detections


def pil_to_cv2(pil_image):
    # Convert PIL Image to NumPy array
    numpy_image = np.array(pil_image)
    # Convert RGB to BGR
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def binarize_image(cv2_image, threshold=128):
    # Convert to grayscale
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    # Apply binary threshold
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def get_empty_cards_numbers(detections: sv.Detections, image: ImageFile) -> list:
    cards_numbers = []
    print(detections)
    for coordinates in detections.xyxy:
        x0, y0, x1, y1 = coordinates
        # cropped_image = image.crop((x0, y0, x1, y1)).convert("L")

        cropped_image = image.crop((x0, y0, x1, y1))
        cv2_image = pil_to_cv2(cropped_image)
        binary_image = binarize_image(cv2_image)

        sv.plot_image(binary_image)
        card_number = infer_client.ocr_image(binary_image)
        cards_numbers.append(card_number)
        print(card_number)

    return cards_numbers


image_file = "imgs/test/enaldinho_18.jpeg"
image = Image.open(image_file)
empty_cards = locate_empty_cards(image)
get_empty_cards_numbers(empty_cards, image)
