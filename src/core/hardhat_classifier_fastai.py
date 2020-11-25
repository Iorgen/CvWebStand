from fastai.vision.all import load_learner
import torch


class Clasp_glasses_classifier:
    def __init__(self):
        if torch.cuda.is_available():
            device_type = 'cuda:0'
        else:
            device_type = 'cpu'

        if device_type == 'cuda:0':
            CPU = False

        model_path = '../../models/hardhat.pkl'
        self.model = load_learner(model_path) # , cpu=CPU

    def predict(self, img_ori):
        # Run inference
        image = img_ori[:, :, [2, 1, 0]]
        pred = self.model.predict(image)[1].tolist()
        clasp, glasses = pred[:2]

        return clasp, glasses

