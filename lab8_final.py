import cv2
import time
import numpy as np

# Задание №1


def modifieded_image():

    image = cv2.imread("images/variant-10.jpg")

    _, modified_img = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # output_path = 'images/modified_variant-10.jpg'
    # cv2.imwrite('images/modified_variant-10.jpg', modified_img)

    cv2.imshow("Original Image", image)
    cv2.imshow("Modified Image", modified_img)


# Задание №2


def track_marker(frame, template):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[1], template.shape[0]

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    return frame


def mark_search():

    template = cv2.imread("ref-point.jpg", cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(0)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_frame = track_marker(frame, template)
        cv2.imshow("Marker Tracking", tracked_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)
        i += 1

    cap.release()


# Задание №3


def track_marker_1(frame, template):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[1], template.shape[0]

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    detected_points = []
    for pt in zip(*locations[::-1]):
        detected_points.append(pt)
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    return frame, detected_points


def mark_search_1():

    template = cv2.imread("ref-point.jpg", cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(0)
    i = 0

    square = 150
    center_x, center_y = square // 2, square // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, detected_points = track_marker_1(frame, template)
        for pt in detected_points:
            mark_center_x = pt[0] + template.shape[1] // 2
            mark_center_y = pt[1] + template.shape[0] // 2

            if (
                mark_center_x >= (frame.shape[1] // 2 - center_x)
                and mark_center_x <= (frame.shape[1] // 2 + center_x)
                and mark_center_y >= (frame.shape[0] // 2 - center_y)
                and mark_center_y <= (frame.shape[0] // 2 + center_y)
            ):

                frame = cv2.flip(frame, 1)

        cv2.imshow("Changing the camera when marking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)
        i += 1

    cap.release()


# Доп. задание


def track_marker_2(frame, template):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[1], template.shape[0]

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    locations = np.where(result >= threshold)

    marker_centers = []
    for pt in zip(*locations[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        center_x = pt[0] + w // 2
        center_y = pt[1] + h // 2
        marker_centers.append((center_x, center_y))

    if not marker_centers:
        return frame, None
    else:
        return frame, marker_centers[0]


def fly(frame, fly, marker_center):

    fly_height, fly_width, _ = fly.shape
    frame_height, frame_width, _ = frame.shape

    x_offset = int(marker_center[0] - fly_width / 2)
    y_offset = int(marker_center[1] - fly_height / 2)

    x1, y1 = max(0, x_offset), max(0, y_offset)
    x2, y2 = min(frame_width, x_offset + fly_width), min(
        frame_height, y_offset + fly_height
    )

    fly_x1 = max(0, -x_offset)
    fly_y1 = max(0, -y_offset)
    fly_x2 = fly_width - max(0, (x_offset + fly_width) - frame_width)
    fly_y2 = fly_height - max(0, (y_offset + fly_height) - frame_height)

    frame[y1:y2, x1:x2] = fly[fly_y1:fly_y2, fly_x1:fly_x2]

    return frame


def mark_search_2():

    template = cv2.imread("ref-point.jpg", cv2.IMREAD_GRAYSCALE)
    fly_image = cv2.imread("fly64.png")

    cap = cv2.VideoCapture(0)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_frame, marker_center = track_marker_2(frame, template)

        if marker_center:
            tracked_frame = fly(tracked_frame, fly_image, marker_center)

        cv2.imshow("Fly on the marker", tracked_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.1)
        i += 1


if __name__ == "__main__":
    # modifieded_image()
    # mark_search()
    # mark_search_1()
    mark_search_2()


cv2.waitKey(0)
cv2.destroyAllWindows()
