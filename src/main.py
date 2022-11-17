import cv2 as cv
import numpy as np
import os
from type_func import *
import matplotlib.pyplot as plt

np.seterr(divide = 'ignore') 

# Current working directory initialization
PATH = os.getcwd()
DATASET_DIR = os.path.join(PATH,"images/")

# Dataset dictionary created
DATASET = {
    "medicine" : init_class_dict(),
    "milk" : init_class_dict(),
    "apple_juice": init_class_dict(),
    "orange_juice": init_class_dict(),
    "orange_juice2": init_class_dict(),
}

test_classes = ["medicine", "milk", "apple_juice", "orange_juice", "orange_juice2"]
test_imgs = ["first", "second", "third", "fourth", "fifth"]

# Homework class for bayesian classification functions
class Homework():
    def __init__(self):
        print("Homework initialization started")

        self.get_images()
        #self.plot_data()

        print("Homework initialized")

    # Dataset initialized
    def get_images(self):
        for i in range(25):
            filename = DATASET_DIR + get_img_name(i)
            file = cv.imread(filename= filename)

            DATASET[get_class_name(i)][get_img_index(i)]["name"] = filename
            DATASET[get_class_name(i)][get_img_index(i)]["img"] = file.T
            calculate_histogram(image = file.T, color_hist = DATASET[get_class_name(i)][get_img_index(i)]["color_hist"])

        # deleted the variables for avoiding memory leak
        del filename
        del file

    def plot_data(self):
        fig = plt.figure(figsize= (12,6))
        count_arr = np.arange(0,768,1, int)

        plt.subplot(311)
        plt.bar(count_arr, DATASET["medicine"]["first"]["color_hist"], 1, color=('black'), alpha=0.5)
        plt.bar(count_arr, DATASET["milk"]["first"]["color_hist"], 1, color=('blue'), alpha=0.5)
        plt.bar(count_arr, DATASET["apple_juice"]["first"]["color_hist"], 1, color=('red'), alpha=0.5)
        plt.bar(count_arr, DATASET["orange_juice"]["first"]["color_hist"], 1, color=('green'), alpha=0.5)
        plt.bar(count_arr, DATASET["orange_juice2"]["first"]["color_hist"], 1, color=('yellow'), alpha=0.5)
        plt.title("Histogram values")
        plt.ylabel("Count")
        
        plt.subplot(312)
        plt.bar(count_arr, ((DATASET["medicine"]["first"]["color_hist"])/(240*320)), 1, color=('black'), alpha=0.5)
        plt.bar(count_arr, ((DATASET["milk"]["first"]["color_hist"])/(240*320)), 1, color=('blue'), alpha=0.5)
        plt.bar(count_arr, ((DATASET["apple_juice"]["first"]["color_hist"])/(240*320)), 1, color=('red'), alpha=0.5)
        plt.bar(count_arr, ((DATASET["orange_juice"]["first"]["color_hist"])/(240*320)), 1, color=('green') , alpha=0.5)
        plt.bar(count_arr, ((DATASET["orange_juice2"]["first"]["color_hist"])/(240*320)), 1, color=('yellow') , alpha=0.5)
        plt.title("PMF")
        plt.ylabel("Probability Masses")


        plt.subplot(313)
        plt.bar(count_arr, create_cdf((DATASET["medicine"]["first"]["color_hist"])/(240*320)), 1, color=('black'), alpha=0.5)
        plt.bar(count_arr, create_cdf((DATASET["milk"]["first"]["color_hist"])/(240*320)), 1, color=('blue'), alpha=0.5)
        plt.bar(count_arr, create_cdf((DATASET["apple_juice"]["first"]["color_hist"])/(240*320)), 1, color=('red'), alpha=0.5)
        plt.bar(count_arr, create_cdf((DATASET["orange_juice"]["first"]["color_hist"])/(240*320)), 1, color=('green'), alpha=0.5)
        plt.bar(count_arr, create_cdf((DATASET["orange_juice2"]["first"]["color_hist"])/(240*320)), 1, color=('yellow'), alpha=0.5)
        plt.title("CDF")
        plt.xlabel("Pixel values multiplied by channels")
        plt.ylabel("Count")

        plt.subplots_adjust(hspace=0.4)
        plt.legend(["medicine", "milk", "apple_juice", "orange_juice", "orange_juice2"])
        plt.show()


    def bayesian_classifier(self, img, img_hist) -> int:
        prob_A = 5 / 25
        prob_A_B = []

        class_count = 5
        for index in range(class_count):
            if index == 0:
                prob_B_A_to_B = self.prob_calculator(img, DATASET["medicine"]["first"]["color_hist"] , img_hist) 
            elif index == 1:
                prob_B_A_to_B = self.prob_calculator(img, DATASET["milk"]["first"]["color_hist"] , img_hist)
            elif index == 2:
                prob_B_A_to_B = self.prob_calculator(img, DATASET["apple_juice"]["first"]["color_hist"] , img_hist)
            elif index == 3:
                prob_B_A_to_B = self.prob_calculator(img, DATASET["orange_juice"]["first"]["color_hist"] , img_hist)
            else:
                prob_B_A_to_B = self.prob_calculator(img, DATASET["orange_juice2"]["first"]["color_hist"] , img_hist)

            prob_A_B.append((prob_B_A_to_B * prob_A) / (240 * 320 * 3))

        print(np.argmax(prob_A_B), "all: ", prob_A_B)

        return np.argmax(prob_A_B)

    def prob_calculator(self, img, class_hist, img_hist):
        prob = 0
        
        for channel in range(len(img)):
            for x in range(len(img[channel])):
                for y in range(len(img[channel][x])):
                    pixel_index = ((channel * 256) + img[channel][x][y])
                    prob = prob + (class_hist[pixel_index] / img_hist[pixel_index])
        return prob



# main function
def main():
    homework = Homework()
    true_count = 0
    for i in range(len(test_classes)):
        for j in range(len(test_imgs)):
            print(test_classes[i], test_imgs[j])
            predicted_class = homework.bayesian_classifier(DATASET[test_classes[i]][test_imgs[j]]["img"], DATASET[test_classes[i]][test_imgs[j]]["color_hist"])
            if predicted_class == i:
                true_count = true_count + 1
    print("acc: ", true_count / 25)

# code initialization
if __name__ == "__main__":
    main()
