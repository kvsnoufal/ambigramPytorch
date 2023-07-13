from fonthelper import *
from glob import glob
import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io
import os
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import os
import easyocr
import config

from utils import *

os.makedirs(config.output_path,exist_ok=True)
os.makedirs(config.write_path,exist_ok=True)
gamma = 1.0
m = getModel()
m = m.to(config.device)


for font_path in config.fonts:
    for txt in config.txts:
        # create svg files from font and text
        font_string_to_svgs(config.output_path, font_path, txt, target_control=target_cp,
                            subdivision_thresh=None)
        
        normalize_letter_size(config.output_path, font_path, txt)

        svg_file = os.path.join(config.output_path,"{}_{}_scaled.svg".format(os.path.basename(font_path)[:-4],txt))
        canvas_width, canvas_height, shapes, shape_groups = \
                pydiffvg.svg_to_scene(svg_file)
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)


        render = pydiffvg.RenderFunction.apply
        img_init = render(canvas_width, # width
                        canvas_height, # height
                        2,   # num_samples_x
                        2,   # num_samples_y
                        0,   # seed
                        None, # bg
                        *scene_args)
        pydiffvg.imwrite(img_init.cpu(), os.path.join(config.write_path,'init.png'), gamma=gamma)
        # enable gradients in the points
        points_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
        # define optimizer
        points_optim = torch.optim.Adam(points_vars, lr=0.1)
        
        img_init = prepare_image(img_init)
        # get target embedding
        text_for_pred = torch.LongTensor(1, 1 + 1).fill_(0).to(device)
        tmp_img = process_for_ocr_model(img_init)
        target_output = m(tmp_img,text_for_pred)

        min_loss =   np.inf
        frames = []
        for t in range(config.iterations):
            points_optim.zero_grad()
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                canvas_width, canvas_height, shapes, shape_groups)
            img = render(canvas_width, # width
                            canvas_height, # height
                            2,   # num_samples_x
                            2,   # num_samples_y
                            0,   # seed
                            None, # bg
                            *scene_args)
            img = prepare_image(img)

            if (t%10==0):
                fpath = os.path.join(config.write_path, '/iter_{}.png'.format(t))
                pydiffvg.imwrite(img.cpu(),fpath, gamma=gamma)
                frame = Image.open(fpath)
                os.remove(fpath)
                frames.append(frame)
            
            # prepare embedding
            target_img = rotate_image(img)
            tmp_img = process_for_ocr_model(img)
            up_embed = m(tmp_img,text_for_pred)

            lossEmb = (up_embed - target_output.detach()).pow(2).mean()




            lossImg = (img - target_img).pow(2).mean()

            loss = lossImg+lossEmb*config.embeddingLossWeight

            loss.backward()

            print('iteration:', t, " loss: ",loss.item())
            
            points_optim.step()
        gifpath = os.path.join(config.write_path,'{}_{}.gif'.format(txt,os.path.basename(font_path[:-4])))
        frames[0].save(gifpath, save_all=True, append_images=frames[1:], duration=200, loop=0)
        pngpath = os.path.join(config.write_path,'{}_{}.png'.format(txt,os.path.basename(font_path[:-4])))
        frames[-1].save(pngpath)