import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from skimage.io import imread
from skimage.filters import threshold_otsu
import sys
import pytesseract
import cv2
from PIL import Image

plate_like_objects = []

filename = sys.argv[1]

import cv2
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# car image -> grayscale image -> binary image
import imutils
car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
print(car_image.shape)

gray_car_image = car_image * 255
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)

# print(label_image.shape[0]) #width of car img

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.04*label_image.shape[0], 0.2*label_image.shape[0], 0.2*label_image.shape[1], 0.6*label_image.shape[1])
# plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
plate_dimensions2 = (0.04*label_image.shape[0], 0.5*label_image.shape[0], 0.2*label_image.shape[1], 0.6*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
# plate_like_objects = []

flag =0

if(flag==0):
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    # plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        if region.area < 50:
            #if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox

        region_height = max_row - min_row
        region_width = max_col - min_col

        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:

            plate_like_objects.append(binary_car_image[min_row:max_row,
                                        min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                                max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                            linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
            Cropped = gray_car_image[min_row:max_row, min_col:max_col]
            # plt.imshow(Cropped)
            # plt.show()
            text = pytesseract.image_to_string(Cropped, config='--psm 11')
            print("Detected Number is:",text)
            # break
            # let's draw a red rectangle over those regions
    # print(plate_like_objects[0])
    plt.show()
    # plt.imshow(Cropped,cmap='gray')

    # Read the number plate
    # text = pytesseract.image_to_string(Cropped, config='--psm 11')
    # print("Detected Number is:",text)

    plt.show()







# print(DetectPlate.plate_like_objects)

# The invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(plate_like_objects[0])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
license_plate = rgb2gray(license_plate)
ax1.imshow(license_plate, cmap="gray")

character_dimensions = (0.5*license_plate.shape[0], 1.0*license_plate.shape[0], 0.00*license_plate.shape[1], 0.4*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter=0
column_list = []

rois = []

for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # draw a red bordered rectangle over the character.
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)
        # plt.imshow(roi)
        # plt.show()
        rois.append(roi)

        # resize the characters to 20X20 and then append each character into the characters list
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)
# print(characters)
plt.show()

for roi in rois:
    roi = np.pad(roi,10)
    roi1 = np.zeros((roi.shape[0],roi.shape[1]*10))
    for i in range(9):
        roi1[:,i*roi.shape[1]:(i+1)*roi.shape[1]] = roi
    roi = roi1
    roi = np.pad(roi,100)
    # roi = roi[::2,::2]
    # roi = 1*(roi==1)
    # print(roi)
    # roi = cv2.bilateralFilter(np.array(roi,dtype='f'),2,1,1)
    roi = np.array(255-roi*255,dtype='f')
    # roi = np.array(1*(roi>50),dtype='f')
    
    # print(1*(roi==1))
    # roi = 1*(roi==1)
    # print(roi)
    text = pytesseract.image_to_string(Image.fromarray(roi), config="--psm 11")
    print(text)

    plt.imshow(roi,cmap='gray')
    plt.show()


################# Add code to detect each character (37 = 26+10+1 classes) in the for loop above #######################

## You have been provided with the base code above
## and just need to add code to detect each character

## To test your code, visualise the image of each character "carefully" and check if your output is matching with your visual observation :)
## It's really easy. I must admit, it's really very easy.
## Ok if INVALID class not added initially, but huge penalty will be added later

## (OPTIONAL) Read the code above and modify the parameters so that it segments correctly

################################ DO NOT EDIT THE CODE BELOW, IT's JUST COMMENTED :) ####################################


# import SegmentCharacters
# import pickle
# print("Loading model...")
# filename = './finalized_model.sav'
# model = pickle.load(open(filename, 'rb'))

# print('Model loaded. Predicting characters of number plate')
# classification_result = []
# for each_character in characters:
#     # converts it to a 1D array
#     plt.imshow(each_character.reshape,cmap='gray')
#     plt.show()
#     print(pytesseract.image_to_string(Image.fromarray(1-each_character)))
#     each_character = each_character.reshape(1, -1);
#     result = model.predict(each_character)
#     classification_result.append(result)

# print('Classification result')
# print(classification_result)

# plate_string = ''
# for eachPredict in classification_result:
#     plate_string += eachPredict[0]
#     # print(eachPredict[0])

# print('Predicted license plate')
# print(plate_string)

# it's possible the characters are wrongly arranged
# since that's a possibility, the column_list will be
# used to sort the letters in the right order

# column_list_copy = column_list[:]
# column_list.sort()
# rightplate_string = ''
# for each in column_list:
#     rightplate_string += plate_string[column_list_copy.index(each)]

# print('License plate')
# print(rightplate_string)