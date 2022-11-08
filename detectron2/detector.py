from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import cv2


class Detector:
    def __init__(self, model_type = "OD"):
        self.cfg = get_cfg()

        #Load model config and pretrained model
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)
        self.image_height = 0
        self.image_width = 0
        self.image_channels = 0

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)

        # Predict and show bounding boxes
        predictions = self.predictor(image)
        metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        viz = Visualizer(image[:, :, ::-1], metadata, instance_mode = ColorMode.IMAGE_BW)
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        class_catalog = metadata.thing_classes
        
        classes = [class_catalog[i] for i in predictions['instances'].pred_classes.numpy()]
        # print("class: ", classes)
        # print("box: ", predictions['instances'].pred_boxes)
        output_boxes = predictions['instances'].pred_boxes.tensor.numpy()
        # print(output_boxes)
        # print("mask: ", predictions['instances'].pred_masks)

        # Show the window in full screen
        self.image_height, self.image_width, self.image_channels = image.shape
        print("Image_h: " + str(self.image_height) + ", Image_w: " + str(self.image_width))
        return predictions, output, output_boxes, classes

