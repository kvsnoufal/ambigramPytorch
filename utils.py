
import torch
from torchvision import transforms
import easyocr
import pydiffvg
grscale_converter = transforms.Compose([transforms.Grayscale(num_output_channels=1)])


def crop_object(image_tensor):
            # Find the coordinates of the object region
    object_coords = torch.nonzero(image_tensor < 0.5, as_tuple=False)
    
    # Determine the bounding box for the object region
    min_x = torch.min(object_coords[:, 1])
    max_x = torch.max(object_coords[:, 1])
    min_y = torch.min(object_coords[:, 2])
    max_y = torch.max(object_coords[:, 2])
    
    # Crop the object region from the image tensor
    cropped_tensor = image_tensor[:, min_x:max_x+1, min_y:max_y+1]
    
    return cropped_tensor

def process_for_ocr_model(img):
    img_init_gr = grscale_converter(img.permute(2,0,1))
    cropped_img_init = crop_object(img_init_gr)
    size=(32,100)
    cropped_img_init = transforms.Resize(size)(cropped_img_init)
    cropped_img_init.sub_(0.5).div_(0.5)
    cropped_img_init = cropped_img_init.view(1, *cropped_img_init.size())
    # print(cropped_img_init)
    tmp_img = cropped_img_init
    # print(tmp_img)
    return tmp_img
def getModel():
    reader = easyocr.Reader(['en'], gpu=True)
    m = reader.recognizer.module
    
    m.Prediction = torch.nn.Identity()
    return m
def rotate_image(image_tensor):
    # Rotate the image tensor 90 degrees clockwise
    rotated_tensor = torch.rot90(image_tensor, k=1, dims=(0, 1))

    # Rotate the image tensor 90 degrees clockwise again
    rotated_tensor = torch.rot90(rotated_tensor, k=1, dims=(0, 1))

    return rotated_tensor
def prepare_image(img):
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    

    # %%
    img = img[:, :, :3]
    return img