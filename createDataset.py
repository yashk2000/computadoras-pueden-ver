import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--label", required = True, help = "Enter label")
ap.add_argument("-p", "--path", required = True, help = "Enter path to output folder")
args = vars(ap.parse_args())

vid = cv2.VideoCapture(0)

cv2.namedWindow("Capture images to build the dataset")

counter = 0

while True:
    ret, frame = vid.read()
    cv2.imshow("Capture images to build the dataset", frame)

    if not ret:
        break

    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("[INFO] Closing")
        break
    
    elif k % 256 == 32:
        sample_name = args["path"] + "/" + args["label"] + "{}.png".format(counter)
        cv2.imwrite(sample_name, frame)
        print("[INFO] {} saved".format(sample_name))
        counter += 1

vid.release()
cv2.destroyAllWindows()