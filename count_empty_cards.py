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


def debug_show_annotations(detections, inference_image):
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


def get_all_images(directory: str) -> list[str]:
    files = []
    for file in os.listdir(directory):
        file_with_path = os.path.join(directory, file)
        if os.path.isfile(file_with_path):
            files.append(file_with_path)
    return files


def locate_empty_cards(image: ImageFile) -> sv.Detections:
    empty_cards = detect_client.infer(inference_input=image, model_id="album-wrkp7/1")
    detections = sv.Detections.from_inference(empty_cards)
    return detections


total_empty_cards = 0
images = get_all_images("imgs/test")
for img in images:
    image = Image.open(img)
    empty_cards = locate_empty_cards(image)
    total_empty_cards += len(empty_cards.xyxy)
    print(f"Empty cards in {img}: {len(empty_cards.xyxy)}")


print(f"Total empty cards: {total_empty_cards}")  # Result: 89, Real Result: 103
