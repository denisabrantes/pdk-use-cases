import torch
import numpy as np
from torchvision import transforms
from ts.torch_handler.object_detector import ObjectDetector
from torch.profiler import ProfilerActivity


class ObjectDetectionHandler(ObjectDetector):

    def __init__(self):
        super(ObjectDetectionHandler, self).__init__()
        self.profiler_args = {
            "activities" : [ProfilerActivity.CPU],
            "record_shapes": True,
        }



    def preprocess(self, data):
        """Preprocess the data, fetches the image from the request body and converts to torch tensor.
        Args:
            data (list): Image to be sent to the model for inference.
        Returns:
            tensor: A torch tensor in correct format for model
        """

        tensor_data = data[0]["data"]
        tensor_shape = data[0]["shape"]
        output = torch.FloatTensor(np.array(tensor_data).reshape(tensor_shape))

        input_img = output.unsqueeze(0)

        return input_img


    # def postprocess(self, data):
    #     """The post process of ObjectDetectionHandler stores the prediction in a list.

    #     Args:
    #         data (tensor): The predicted output from the Inference is passed
    #         to the post-process function
    #     Returns:
    #         list : A list with a tensor for the mask is returned
    #     """
    #     return data.tolist()
