"""
Contains class that runs inferencing
"""
import torch
import numpy as np
import torch.nn.functional as F

from networks.RecursiveUNet import UNet
from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """

        reshaped = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))

        return self.single_volume_inference(reshaped)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []
        for slc in volume:
            indata = torch.tensor(slc).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(self.device)
            rawp = self.model(indata)
            amax = torch.argmax(rawp, dim=1)
            pred = amax.squeeze().type(torch.LongTensor).cpu()
            slices.append(pred.numpy())


        return np.array(slices)

