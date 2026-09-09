import re
import importlib


try:
    Image = importlib.import_module("PIL.Image")
    ImageEnhance = importlib.import_module("PIL.ImageEnhance")
    ImageFilter = importlib.import_module("PIL.ImageFilter")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The 'Pillow' package is required. Install it with: "
        "python -m pip install Pillow"
    ) from exc


try:
    pytesseract = importlib.import_module("pytesseract")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "The 'pytesseract' package is required. Install it with: "
        "python -m pip install pytesseract"
    ) from exc


# Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def preprocess_image(image):
    image = image.convert("L")

    image = ImageEnhance.Contrast(image).enhance(2.0)

    image = ImageEnhance.Sharpness(image).enhance(2.0)

    image = image.filter(ImageFilter.SHARPEN)

    return image


def extract_text(image_path):
    image = Image.open(image_path)

    processed_image = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed_image,
        config="--psm 6"
    )

    return text.strip()


def detect_document_type(text):
    text_upper = text.upper()

    if (
        "PASSPORT" in text_upper
        or "MRZ" in text_upper
        or "P<" in text_upper
    ):
        return "passport"

    if "VISA" in text_upper:
        return "visa"

    if (
        "DRIVING" in text_upper
        or "DRIVER LICENSE" in text_upper
        or "DRIVING LICENCE" in text_upper
    ):
        return "driving_license"

    if (
        "NATIONAL ID" in text_upper
        or "IDENTITY CARD" in text_upper
    ):
        return "national_id"

    return "unknown"


def extract_document_number(text):
    patterns = [
        r"(?:PASSPORT\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9]{6,15})",

        r"(?:DOCUMENT\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9]{6,20})",

        r"(?:VISA\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9]{6,20})",

        r"(?:LICENSE|LICENCE)\s*(?:NO|NUMBER)?\s*[:\-]?\s*([A-Z0-9]{6,20})",

        r"(?:ID\s*(?:NO|NUMBER))\s*[:\-]?\s*([A-Z0-9]{6,20})"
    ]

    text_upper = text.upper()

    for pattern in patterns:
        match = re.search(
            pattern,
            text_upper
        )

        if match:
            return match.group(1)

    return None


def extract_dates(text):
    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",

        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",

        r"\b\d{2}\.\d{2}\.\d{4}\b"
    ]

    dates = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text
        )

        dates.extend(matches)

    return list(dict.fromkeys(dates))


def extract_name(text):
    lines = text.splitlines()

    blocked_words = [
        "PASSPORT",
        "VISA",
        "NAME",
        "SURNAME",
        "NATIONALITY",
        "DATE",
        "DOB",
        "SEX",
        "GENDER",
        "ADDRESS",
        "LICENSE",
        "LICENCE",
        "IDENTITY",
        "CARD",
        "NUMBER",
        "NO",
        "VALID",
        "EXPIRY"
    ]

    for line in lines:
        line = line.strip()

        if not line:
            continue

        clean_line = re.sub(
            r"[^A-Za-z ]",
            "",
            line
        ).strip()

        words = clean_line.split()

        if 2 <= len(words) <= 5:
            upper_line = clean_line.upper()

            if not any(
                word in upper_line
                for word in blocked_words
            ):
                return clean_line

    return None


def analyze_ocr(image_path):
    text = extract_text(image_path)

    document_type = detect_document_type(text)

    document_number = extract_document_number(text)

    dates = extract_dates(text)

    name = extract_name(text)

    return {
        "text": text,

        "document_type": document_type,

        "document_number": document_number,

        "name": name,

        "dates": dates,

        "ocr_status": (
            "SUCCESS"
            if text
            else "NO TEXT DETECTED"
        )
    }