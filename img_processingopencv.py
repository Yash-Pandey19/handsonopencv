import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_image():
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.rectangle(img, (200, 200), (350, 300), 255, -1)
    for i in range(0, 400, 20):
        cv2.line(img, (i, 350 + int(10 * np.sin(i / 20.0 * np.pi))), (i + 20, 350 + int(10 * np.sin((i + 20) / 20.0 * np.pi))), 255, 2)
    return img

def process_image(img):
    binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(img, 100, 200)
    return binary_img, edges

def show_histogram(img):
    plt.hist(img.ravel(), bins=256, range=[0, 256])
    plt.title("Histogram")
    plt.show()

def main():
    img = create_image()
    binary_img, edges = process_image(img)
    show_histogram(img)
    cv2.imshow("Original", img)
    cv2.imshow("Binary", binary_img)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
