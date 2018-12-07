import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from transforms import *
import numpy as np



__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=40):
        super(AlexNet, self).__init__()
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.crop_size = 224
        self.scale_size = 256
        self.hSize = 128
        self.viewNum = 20
        self.layerNum = 1
        self.indexmat = np.load('vcand_case2.npy')
        self.rowmat = np.load('rowmat.npy')
        self.nClasses = num_classes
        #self.indexmat = np.load('sequence.npy') - 1
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(nn.Linear(self.hSize*self.viewNum*3, self.nClasses))
        self.gru = nn.GRU(4096,self.hSize,self.layerNum,batch_first=True,bidirectional=False)
    def forward(self, x):
        inputsz = x.size()
        x = x.view((inputsz[0]*inputsz[1]//3,3,inputsz[2],inputsz[3]))
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        sz = x.size()
        x = x.view(sz[0]//self.viewNum,self.viewNum,sz[1])
        scores_all = torch.zeros(sz[0]//self.viewNum, self.viewNum*3, self.nClasses).type(torch.cuda.FloatTensor)
        hn_all = torch.zeros(sz[0]//self.viewNum, 60,self.hSize).type(torch.cuda.FloatTensor)
        hx = torch.zeros(self.layerNum, sz[0]//self.viewNum, self.hSize).type(torch.cuda.FloatTensor)

        for i in range(self.indexmat.shape[0]):
            indices = self.indexmat[i]
            x_tmp = x[:,indices,:] 
            output,hn = self.gru(x_tmp.view(-1,20,4096),hx)
            hn_all[:,i,:] = output[:,5,:]#self.fc(hn.contiguous().view(sz[0]//self.viewNum,-1))

        for i in range(self.indexmat.shape[0]):
            rows = self.rowmat[i]
            scores_all[:,i,:] = self.fc(hn_all[:,rows,:].contiguous().view(sz[0]//self.viewNum,-1)) 

        scores,index = torch.max(scores_all,1)
        return scores



    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875])])


def alexnet(pretrained=False, num_classes=40):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(num_classes)
    if pretrained:
            model_dict = model.state_dict()
            checkpoint = model_zoo.load_url(model_urls['alexnet']);
            base_dict = {k:v for k,v in list(checkpoint.items())}
            pretrained_dict = {k for k in model_dict}
            pretrained_dict = {k: v for k, v in base_dict.items() if k in model_dict}
            for x in pretrained_dict:
                print(x)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    return model
