import cv2
import os
import numpy as np

FORMAT_FOLDER = "backend/formate"


def compare_images(uploaded_path, reference_path):

    uploaded = cv2.imread(uploaded_path)
    reference = cv2.imread(reference_path)

    if uploaded is None or reference is None:
        return 0.0

    # Standard size
    uploaded = cv2.resize(uploaded, (600, 800))
    reference = cv2.resize(reference, (600, 800))

    # Grayscale
    uploaded_gray = cv2.cvtColor(
        uploaded,
        cv2.COLOR_BGR2GRAY
    )

    reference_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    # Blur to reduce differences caused by
    # photo, text and scan quality
    uploaded_blur = cv2.GaussianBlur(
        uploaded_gray,
        (9, 9),
        0
    )

    reference_blur = cv2.GaussianBlur(
        reference_gray,
        (9, 9),
        0
    )

    # Edge structure
    uploaded_edges = cv2.Canny(
        uploaded_blur,
        50,
        150
    )

    reference_edges = cv2.Canny(
        reference_blur,
        50,
        150
    )

    # Structural difference
    edge_difference = cv2.absdiff(
        uploaded_edges,
        reference_edges
    )

    edge_difference_value = (
        np.mean(edge_difference)
    )

    edge_score = 100 - (
        edge_difference_value / 255 * 100
    )

    edge_score = max(
        0,
        min(100, edge_score)
    )

    # Large-scale layout comparison
    uploaded_small = cv2.resize(
        uploaded_gray,
        (30, 40)
    )

    reference_small = cv2.resize(
        reference_gray,
        (30, 40)
    )

    pixel_difference = cv2.absdiff(
        uploaded_small,
        reference_small
    )

    pixel_difference_value = (
        np.mean(pixel_difference)
    )

    layout_score = 100 - (
        pixel_difference_value / 255 * 100
    )

    layout_score = max(
        0,
        min(100, layout_score)
    )

    # Final format match
    format_match = (
        edge_score * 0.70
        +
        layout_score * 0.30
    )

    format_match = max(
        0,
        min(100, format_match)
    )

    return round(
        format_match,
        2
    )


def calculate_risk(match_percentage):

    risk_score = 100 - match_percentage

    if risk_score <= 15:
        risk_level = "LOW"

    elif risk_score <= 30:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "risk_score": round(
            risk_score,
            2
        ),
        "risk_level": risk_level
    }


def find_reference(document_type):

    reference_path = os.path.join(
        FORMAT_FOLDER,
        document_type,
        "reference.png"
    )

    if os.path.exists(reference_path):
        return reference_path

    return None