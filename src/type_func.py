import numpy as np

def init_class_dict():
    return {
        "first": init_image_dict(),
        "second": init_image_dict(),
        "third": init_image_dict(),
        "fourth": init_image_dict(),
        "fifth": init_image_dict(),
    }

def init_image_dict():
    return {
        "name": str,
        "img": np.array([]),
        "color_hist": np.zeros(256*3, dtype=int)
    }

def get_img_index(index:int) -> str:
    if index % 5 == 0:
        return "first"
    elif index % 5 == 1:
        return "second"
    elif index % 5 == 2:
        return "third"
    elif index % 5 == 3:
        return "fourth"
    else:
        return "fifth"

def get_class_name(index:int) -> str:
    if index < 5:
        return "medicine"
    elif index < 10:
        return "milk"
    elif index < 15:
        return "apple_juice"
    elif index < 20:
        return "orange_juice"
    elif index < 25:
        return "orange_juice2"


def get_img_name(index:int) -> str:
    if index < 5:
       return "image00"+ str(index+5) + ".jpg"
    else:
        return "image0"+ str(index+5) + ".jpg"


def calculate_histogram(image: np.ndarray, color_hist: np.ndarray) -> None:
    for channel in range(len(image)):
        for x in range(len(image[channel])):
            for y in range(len(image[channel][x])):
                hist_block = (channel * 256) + image[channel][x][y]
                #print("channel", channel, "x", x, "y", y, "hist_block", hist_block , "color", color_hist[hist_block])
                color_hist[hist_block] = color_hist[hist_block] + 1
    
    del hist_block
    
def create_cdf(pmf):
    cdf = np.zeros(768)
    for i in range(len(pmf)):
        if i == 0 or i == 256 or i == 512:
            cdf[i] = pmf[i]
        else:
            cdf[i] = cdf[i-1] + pmf[i]
    return cdf
    