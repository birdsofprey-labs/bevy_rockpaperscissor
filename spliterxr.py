import cv2
import sys, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import OpenEXR
import Imath

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

Image.MAX_IMAGE_PIXELS = 933120000

def save_exr(out_filename, image):
   

    # Create an OpenEXR header
    header = OpenEXR.Header(image.shape[1], image.shape[0])

    out_file = OpenEXR.OutputFile(out_filename, header)
    #out_file.writeHeader(header)
    out_file.writePixels({'R':image, 'G':image, 'B':image})

    # Close the output file
    out_file.close()


def make_tile(id, itype,xy, hw):
    max_size = 512
    image = img
    
    pad = 0
    
    cropped = image
    if cropped.width > max_size:
        cropped = cropped.crop((xy[0]-pad,xy[1]-pad,xy[0]+hw[0]+pad,xy[1]+hw[1]+pad))
        cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)
        cropped = cropped.resize((max_size+pad,max_size+pad))
        #cropped = cropped.crop((pad, pad, max_size-pad, max_size-pad))
    else:
        cropped = cropped.crop((xy[0]-pad,xy[1]-pad,xy[0]+hw[0]+pad,xy[1]+hw[1]+pad))
        cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)

    if itype == 'hgt':
        h = np.array(cropped)
        h = h.astype("float32")
        save_exr(f'../assets/tiles/{itype}_{faceid}_{id}.exr', h)
    else:
        cropped.save(f'../assets/tiles/{itype}_{faceid}_{id}.png') 

image_filename = sys.argv[1].strip()
save_as = sys.argv[2].strip()
faceid = sys.argv[3].strip()

if save_as == 'hgt':
    img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
    img = img[:,:,0] 
    img = Image.fromarray(img) 
else:
    img = Image.open(image_filename)
    
print('processing', image_filename)
width, height = img.size

max_depth = 4
def split(depth, id, xy, wh):
    # = depth + 2 
    D = wh // 2
    NW = xy
    NE = xy + np.array([wh[0] // 2, 0])
    SW = xy + np.array([0, wh[1] // 2])
    SE = xy + wh / 2
    id *= 4

    print(depth, id, xy, wh, save_as)
 

    if depth < max_depth:
        split(depth+1, id+1, NW, D)
        split(depth+1, id+2, NE, D)
        split(depth+1, id+3, SW, D)
        split(depth+1, id+4, SE, D)
        
    make_tile(id+1, save_as, NW, D)
    make_tile(id+2, save_as, NE, D)
    make_tile(id+3, save_as, SW, D)
    make_tile(id+4, save_as, SE, D)

    