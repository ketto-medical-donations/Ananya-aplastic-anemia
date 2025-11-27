import streamlit as st
import cv2
import numpy as np

st.title("RBC Detector (Hough Circles + Oval Detection)")
st.write("Detects circular and oval RBCs while excluding purple nuclei.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    output = img.copy()

    # Hough circle detection (original code)
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

    purple_lower = np.array([135, 120, 40])
    purple_upper = np.array([160, 255, 160])

    valid_count = 0

    # --- Draw circular RBCs ---
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:

            # ROI for purple check
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(img.shape[1], x + r)
            y2 = min(img.shape[0], y + r)
            roi_hsv = hsv[y1:y2, x1:x2]

            purple_mask = cv2.inRange(roi_hsv, purple_lower, purple_upper)
            purple_pixels = np.sum(purple_mask > 0)

            if purple_pixels > 0:
                continue

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            valid_count += 1

    # ---------- OVAL RBC DETECTION (NEW) ----------
    # Use contours to catch oval/elliptical RBCs
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 20:
            continue

        area = cv2.contourArea(cnt)
        if not (200 < area < 3000):
            continue  # remove tiny dots or giant artifacts

        # Fit ellipse if possible
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse

            aspect_ratio = MA / ma if ma != 0 else 1

            # Accept elliptical RBC shapes (ovalocytes)
            if 0.3 < aspect_ratio < 3.0:

                # Purple exclusion on ellipse region
                x1 = int(max(0, cx - ma/2))
                y1 = int(max(0, cy - MA/2))
                x2 = int(min(img.shape[1], cx + ma/2))
                y2 = int(min(img.shape[0], cy + MA/2))

                roi_hsv = hsv[y1:y2, x1:x2]
                purple_mask = cv2.inRange(roi_hsv, purple_lower, purple_upper)
                purple_pixels = np.sum(purple_mask > 0)

                if purple_pixels == 0:
                    cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                    valid_count += 1
    # ----------------------------------------------

    # Convert to RGB for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.image(output_rgb, caption="RBC Detection", use_column_width=True)

    _, png_img = cv2.imencode(".png", output_rgb)
    st.download_button("Download Processed Image", png_img.tobytes(), "rbc_filtered.png")
