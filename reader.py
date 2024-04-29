import cv2
import numpy as np

# Load the OMR image, answer area image, answer field image, and dot image
omr_image = cv2.imread('omr.jpg')
answer_area_image = cv2.imread('omr_answer_area.jpg')
answer_field_image = cv2.imread('omr_answer_field.jpg')
a = cv2.imread('a.jpg')
b = cv2.imread('b.jpg')
c = cv2.imread('c.jpg')
d = cv2.imread('d.jpg')

# Convert images to grayscale
gray_omr = cv2.cvtColor(omr_image, cv2.COLOR_BGR2GRAY)
gray_answer_area = cv2.cvtColor(answer_area_image, cv2.COLOR_BGR2GRAY)
gray_answer_field = cv2.cvtColor(answer_field_image, cv2.COLOR_BGR2GRAY)
gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
gray_c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
gray_d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

# Perform template matching for answer areas
result = cv2.matchTemplate(gray_omr, gray_answer_area, cv2.TM_CCOEFF_NORMED)

# Set a threshold for template matching results
threshold = 0.8
loc = np.where(result >= threshold)

# Draw rectangles around the matched answer areas
w, h = gray_answer_area.shape[::-1]
for pt in zip(*loc[::-1]):
    cv2.rectangle(omr_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    # Extract the region of interest (ROI) for each answer area
    roi = gray_omr[pt[1]:pt[1] + h, pt[0]:pt[0] + w]

    # Perform template matching for answer fields within the ROI
    field_result = cv2.matchTemplate(
        roi, gray_answer_field, cv2.TM_CCOEFF_NORMED)

    # Set a threshold for template matching results within the ROI
    field_threshold = 0.57
    field_loc = np.where(field_result >= field_threshold)

    i = 1
    # Draw rectangles around the matched answer fields within the ROI
    field_w, field_h = gray_answer_field.shape[::-1]
    prev_answer_fields = []  # Keep track of previously detected answer fields
    for field_pt in zip(*field_loc[::-1]):
        field_x = pt[0] + field_pt[0]
        field_y = pt[1] + field_pt[1]

        # Check if the current answer field overlaps with any previously detected answer field
        overlap = False
        for prev_field in prev_answer_fields:
            prev_x, prev_y = prev_field[0], prev_field[1]
            prev_w, prev_h = prev_field[2], prev_field[3]
            if field_x < prev_x + prev_w and field_x + field_w > prev_x and field_y < prev_y + prev_h and field_y + field_h > prev_y:
                overlap = True
                break

        if overlap:
            continue

        cv2.rectangle(omr_image, (field_x, field_y), (field_x +
                      field_w, field_y + field_h), (0, 0, 255), 2)

        # Match answer field ROIs with options and print the one with the highest matching value
        roi_a = roi[field_pt[1]:field_pt[1] + field_h,
                    field_pt[0]:field_pt[0] + field_w]
        roi_b = roi[field_pt[1]:field_pt[1] + field_h,
                    field_pt[0]:field_pt[0] + field_w]
        roi_c = roi[field_pt[1]:field_pt[1] + field_h,
                    field_pt[0]:field_pt[0] + field_w]
        roi_d = roi[field_pt[1]:field_pt[1] + field_h,
                    field_pt[0]:field_pt[0] + field_w]

        # Resize template images to match the size of the ROI
        gray_a_resized = cv2.resize(gray_a, (field_w, field_h))
        gray_b_resized = cv2.resize(gray_b, (field_w, field_h))
        gray_c_resized = cv2.resize(gray_c, (field_w, field_h))
        gray_d_resized = cv2.resize(gray_d, (field_w, field_h))

        match_values = []
        try:
            match_values.append(cv2.matchTemplate(
                roi_a, gray_a_resized, cv2.TM_CCOEFF_NORMED))
            match_values.append(cv2.matchTemplate(
                roi_b, gray_b_resized, cv2.TM_CCOEFF_NORMED))
            match_values.append(cv2.matchTemplate(
                roi_c, gray_c_resized, cv2.TM_CCOEFF_NORMED))
            match_values.append(cv2.matchTemplate(
                roi_d, gray_d_resized, cv2.TM_CCOEFF_NORMED))
        except cv2.error as e:
            print(f"Error: {e}")
            continue

        max_match_value = max(match_values)
        max_index = match_values.index(max_match_value)
        options = ['A', 'B', 'C', 'D']
        answer = options[max_index]

        # Print the answer
        print(f"Answer {i}:", answer)
        i += 1

        # Add the current answer field to the list of previously detected answer fields
        prev_answer_fields.append((field_x, field_y, field_w, field_h))

# Display the image with drawn rectangles
cv2.imshow('OMR Image', omr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
