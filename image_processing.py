import cv2
import numpy as np
import config as cfg


def hough_detector(image):
    maybe_signs = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, minDist=20,
                               dp=1, param1=30, param2=60, minRadius=10, maxRadius=250)
    if circles is not None:
        circles = circles.astype(int)
        for circle in circles[0, :]:
            if circle[2] != 0:
                sign = image[max(0, circle[1] - circle[2]):circle[1] + circle[2],
                             max(0, circle[0] - circle[2]):circle[0] + circle[2]]
                sign = cv2.resize(sign, (cfg.img_size, cfg.img_size))
                maybe_signs.append(sign)
    return maybe_signs


def rotate_img(img):
    angle = np.random.randint(-15, 15)
    h, w, c = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def external_random_color(img):
    h, w, c = img.shape
    rand_color = [np.random.randint(255), np.random.randint(
        255), np.random.randint(255)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, minDist=20,
                               dp=1, param1=30, param2=60, minRadius=50, maxRadius=1000)
    largest_circle, r_max = None, 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if r > r_max:
                r_max = r
                largest_circle = (x, y, r)
        for x in range(w):
            for y in range(h):
                if (x-largest_circle[0])**2 + (y-largest_circle[1])**2 > largest_circle[2]**2:
                    img[x, y] = np.array(rand_color)
    return img


def add_noise(img):
    h, w, c = img.shape
    noise = np.random.rand(h, w, c)*np.random.randint(2, 10)
    noise = noise.astype(int)
    return np.clip(img + noise, 0, 255)


def change_brightness(img, a, b):
    return np.clip(a*img + b, 0, 255).astype(int)


def change_perspective(img):
    h, w, c = img.shape
    a = min(w, h)//10+1
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0+np.random.randint(-a, a), 0+np.random.randint(-a, a)],
                       [w-np.random.randint(-a, a), 0 +
                        np.random.randint(-a, a)],
                       [0+np.random.randint(-a, a), h -
                        np.random.randint(-a, a)],
                       [w-np.random.randint(-a, a), h-np.random.randint(-a, a)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h))


def add_blur(img):
    kernel = np.random.randint(2)*2+1
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def random_transformations(img):

    new_img = cv2.resize(img, (42, 42))

    new_img = external_random_color(new_img)

    if np.random.randint(5) < 4:
        new_img = rotate_img(new_img)

    if np.random.randint(3) == 0:
        new_img = add_blur(new_img)

    if np.random.randint(2) == 0:
        new_img = change_perspective(new_img)
    new_img = change_brightness(
        new_img, np.random.uniform(0.6, 1.0), -np.random.randint(50))

    new_img = add_noise(new_img)

    return new_img
