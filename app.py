import streamlit as st
import cv2
import numpy as np

st.title("RBC Detector (Hough Circles)")
st.write("Upload an RBC smear image to detect and count circular RBCs (excluding purple nuclei).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    output = img.copy()

    # Hough circle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=50,
        param2=30,
        minRadius=12,
        maxRadius=35
    )

    # Convert to HSV for purple nucleus detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ----- DARK PURPLE NUCLEUS HSV RANGE (updated) -----
    # Matches ONLY deep purple WBC nuclei, not light lavender
    purple_lower = np.array([135, 120, 40])
    purple_upper = np.array([160, 255, 160])
    # ----------------------------------------------------

    valid_count = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:

            # ROI for the circle
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(img.shape[1], x + r)
            y2 = min(img.shape[0], y + r)

            roi_hsv = hsv[y1:y2, x1:x2]

            # Mask for dark purple nuclei
            purple_mask = cv2.inRange(roi_hsv, purple_lower, purple_upper)
            purple_pixels = np.sum(purple_mask > 0)

            # Skip circle if dark purple is found
            if purple_pixels > 0:
                continue

            # Otherwise draw the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 255, 0), 3)
            valid_count += 1

    # Convert to RGB for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.subheader(f"Detected RBCs (excluding dark purple nuclei): {valid_count}")
    st.image(output_rgb, caption="Filtered RBC Detection", use_column_width=True)

    # Download processed image button
    _, png_img = cv2.imencode(".png", output_rgb)
    st.download_button(
        "Download Processed Image",
        png_img.tobytes(),
        "rbc_filtered.png",
        "image/png"
    )
