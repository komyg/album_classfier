import os
from dotenv import load_dotenv
from PIL import Image
from PIL.ImageFile import ImageFile
from inference_sdk import InferenceHTTPClient
import supervision as sv

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


def get_empty_cards_numbers(detections: sv.Detections, image: ImageFile) -> list:
    cards_numbers = []
    print(detections)
    for coordinates in detections.xyxy:
        x0, y0, x1, y1 = coordinates
        cropped_image = image.crop((x0, y0, x1, y1)).convert("L")
        sv.plot_image(cropped_image)
        card_number = infer_client.ocr_image(cropped_image)
        cards_numbers.append(card_number)
        print(card_number)

    return cards_numbers


image_file = "imgs/test/enaldinho_16.jpeg"
image = Image.open(image_file)
empty_cards = locate_empty_cards(image)
get_empty_cards_numbers(empty_cards, image)
