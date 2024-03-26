# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet

# %%
model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}

# %%
def get_model(pretrained=None):
    if pretrained == 'backbone':
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        model = FasterRCNN(backbone,num_classes=3,min_size=300, max_size=600)
    elif pretrained == 'coco':
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        model = FasterRCNN(backbone,num_classes=91,min_size=300, max_size=600)
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'], progress=True)
        model.load_state_dict(state_dict)
        num_classes = 3
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    else:
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        model = FasterRCNN(backbone,num_classes=3,min_size=300, max_size=600)
    
    return model
