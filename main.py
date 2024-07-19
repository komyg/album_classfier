import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key=os.environ["ROBOFLOW_API_KEY"]
)

image_file = "poster_2.jpeg"
image = cv2.imread(image_file)

result = CLIENT.infer(inference_input=image, model_id="album-wrkp7/1")

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(result)

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)

# print(result)
