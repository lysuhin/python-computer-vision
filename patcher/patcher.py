
import cv2
import glob

source_folder = "./data/"   # your path
dest_folder =   "./destination" # your destination
image_names = glob.glob(source_folder + '*.*')

CROPSIZE = 36
refPt = []
cropping = False
counter = 0


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, counter
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x - CROPSIZE / 2, y - CROPSIZE / 2), (x + CROPSIZE / 2, y + CROPSIZE / 2)]
        cropping = False
        cropped = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imwrite(dest_folder + image_name.split('/')[-1].split('.')[0] + '-' + str(counter) + ".png", cropped)
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 1)
        counter += 1
    cv2.imshow("image", image)


for j, image_name in enumerate(sorted(image_names)):
    try:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print "(%d/%d) Image read: %s" % (j+1, len(image_names), image_name)
    except:
        print "Error in reading %s" %image_name

    if image == None:
        print "Error in image: empty file"

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
        if key == ord("n"):
            counter = 0
            break
        if key == ord('q'):
            exit()

cv2.destroyAllWindows()
