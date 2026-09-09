import os
import re
import uuid
from datetime import datetime

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

# ============================================================
# BLACK SECURITY
# HEXA NOVA
# AI-Based Fake Identity & Document Screening System
# ============================================================


app = FastAPI(
    title="BLACK SECURITY",
    version="4.0",
    description="Automated document screening prototype"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


try:

    import pytesseract

    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

except ImportError:

    pytesseract = None


# ============================================================
# DOCUMENT RULES
# ============================================================

DOCUMENT_RULES = {

    "passport": {

        "keywords": [
            "passport",
            "nationality",
            "surname",
            "given name",
            "date of birth",
            "sex",
            "date of expiry"
        ],

        "fields": [
            "name",
            "dob",
            "document_number",
            "nationality",
            "gender",
            "expiry"
        ],

        "mrz": True
    },


    "visa": {

        "keywords": [
            "visa",
            "valid",
            "entry",
            "stay",
            "passport"
        ],

        "fields": [
            "name",
            "document_number",
            "visa_type",
            "expiry",
            "entry"
        ],

        "mrz": False
    },


    "national_id": {

        "keywords": [
            "identity",
            "identification",
            "national",
            "date of birth",
            "dob"
        ],

        "fields": [
            "name",
            "dob",
            "document_number",
            "gender"
        ],

        "mrz": False
    },


    "driving_license": {

        "keywords": [
            "driving",
            "driver",
            "licence",
            "license",
            "date of birth"
        ],

        "fields": [
            "name",
            "dob",
            "document_number",
            "expiry"
        ],

        "mrz": False
    }

}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    height, width = image.shape[:2]


    # Resize small images
    if width < 1400:

        scale = 1400 / width

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


    processed = cv2.convertScaleAbs(
        blur,
        alpha=1.35,
        beta=8
    )


    return processed


# ============================================================
# OCR
# ============================================================

def extract_text(image):

    if pytesseract is None:

        return ""


    try:

        original_text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )


        processed = preprocess_image(
            image
        )


        processed_text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )


        original_text = original_text.strip()

        processed_text = processed_text.strip()


        # Choose longer OCR result
        if len(processed_text) > len(original_text):

            return processed_text

        return original_text


    except Exception:

        return ""


# ============================================================
# NORMALIZE OCR
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = text.replace(
        "\n",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# AUTOMATIC DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):

    normalized = normalize_text(
        text
    )


    scores = {}


    for doc_type, rule in DOCUMENT_RULES.items():

        score = 0

        for keyword in rule["keywords"]:

            if keyword.lower() in normalized:

                score += 1


        # Strong passport indicator
        if doc_type == "passport":

            if "passport" in normalized:
                score += 4

            if "nationality" in normalized:
                score += 2


        # Strong visa indicator
        if doc_type == "visa":

            if "visa" in normalized:
                score += 5

            if "entry" in normalized:
                score += 2


        # Driving license indicators
        if doc_type == "driving_license":

            if "driving" in normalized:
                score += 3

            if "license" in normalized:
                score += 2

            if "licence" in normalized:
                score += 2


        # National ID indicators
        if doc_type == "national_id":

            if "identity" in normalized:
                score += 3

            if "identification" in normalized:
                score += 3

            if "national" in normalized:
                score += 2


        scores[doc_type] = score


    detected = max(
        scores,
        key=scores.get
    )


    highest = scores[detected]

    total = sum(
        scores.values()
    )


    if total == 0:

        confidence = 0

    else:

        confidence = round(
            (highest / total) * 100,
            2
        )


    # If nothing meaningful detected
    if highest == 0:

        detected = "unknown"

        confidence = 0


    return (
        detected,
        confidence,
        scores
    )


# ============================================================
# FIELD EXTRACTION
# ============================================================

def extract_fields(text):

    fields = {

        "name": None,

        "dob": None,

        "document_number": None,

        "nationality": None,

        "gender": None,

        "expiry": None,

        "visa_type": None,

        "entry": None

    }


    # --------------------------------------------------------
    # DATE PATTERN
    # --------------------------------------------------------

    date_pattern = (
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{4}[/-]\d{1,2}[/-]\d{1,2})"
    )


    dob_match = re.search(
        r"(date\s+of\s+birth|dob)"
        r"\s*[:\-]?\s*" +
        date_pattern,
        text,
        re.I
    )


    if dob_match:

        fields["dob"] = dob_match.group(
            dob_match.lastindex
        )


    # --------------------------------------------------------
    # EXPIRY
    # --------------------------------------------------------

    expiry_match = re.search(
        r"(date\s+of\s+expiry|expiry|expires|valid\s+until)"
        r"\s*[:\-]?\s*" +
        date_pattern,
        text,
        re.I
    )


    if expiry_match:

        fields["expiry"] = expiry_match.group(
            expiry_match.lastindex
        )


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_match = re.search(
        r"(name|full\s+name|given\s+name)"
        r"\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,60})",
        text,
        re.I
    )


    if name_match:

        fields["name"] = (
            name_match.group(2)
            .strip()
        )


    # --------------------------------------------------------
    # PASSPORT NUMBER
    # --------------------------------------------------------

    passport_match = re.search(
        r"(passport\s*(?:no|number)?|document\s*(?:no|number)?)"
        r"\s*[:#\-]?\s*([A-Z0-9]{6,15})",
        text,
        re.I
    )


    if passport_match:

        fields["document_number"] = (
            passport_match.group(2)
        )


    # --------------------------------------------------------
    # VISA NUMBER
    # --------------------------------------------------------

    visa_number_match = re.search(
        r"(visa\s*(?:no|number)?)"
        r"\s*[:#\-]?\s*([A-Z0-9]{5,20})",
        text,
        re.I
    )


    if visa_number_match:

        fields["document_number"] = (
            visa_number_match.group(2)
        )


    # --------------------------------------------------------
    # NATIONALITY
    # --------------------------------------------------------

    nationality_match = re.search(
        r"nationality\s*[:\-]?\s*"
        r"([A-Za-z]{2,30})",
        text,
        re.I
    )


    if nationality_match:

        fields["nationality"] = (
            nationality_match.group(1)
        )


    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_match = re.search(
        r"(sex|gender)\s*[:\-]?\s*"
        r"(male|female|m|f|other)",
        text,
        re.I
    )


    if gender_match:

        fields["gender"] = (
            gender_match.group(2)
        )


    # --------------------------------------------------------
    # VISA TYPE
    # --------------------------------------------------------

    visa_type_match = re.search(
        r"(visa\s*type|type\s*of\s*visa)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z ]{2,30})",
        text,
        re.I
    )


    if visa_type_match:

        fields["visa_type"] = (
            visa_type_match.group(2)
            .strip()
        )


    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    entry_match = re.search(
        r"(entry|entries)"
        r"\s*[:\-]?\s*"
        r"(single|double|multiple|multiple\s+entry|single\s+entry)",
        text,
        re.I
    )


    if entry_match:

        fields["entry"] = (
            entry_match.group(2)
            .strip()
        )


    # --------------------------------------------------------
    # MRZ FALLBACK
    # --------------------------------------------------------

    lines = text.splitlines()

    mrz_lines = []

    for line in lines:

        clean = line.strip()

        if (
            len(clean) >= 25
            and (
                "<<" in clean
                or clean.startswith("P<")
            )
        ):

            mrz_lines.append(
                clean
            )


    if mrz_lines:

        # MRZ document number
        mrz_number = re.search(
            r"P<[A-Z<]{1,3}([A-Z0-9]{6,12})",
            mrz_lines[0].replace(" ", "")
        )

        if mrz_number and not fields["document_number"]:

            fields["document_number"] = (
                mrz_number.group(1)
            )


    return fields


# ============================================================
# MRZ CHECK
# ============================================================

def detect_mrz(text):

    normalized = text.replace(
        " ",
        ""
    ).upper()


    patterns = [

        "P<",

        "<<<<<<",

        "PASSPORT"

    ]


    return any(
        x in normalized
        for x in patterns
    )


# ============================================================
# DOCUMENT CHECKS
# ============================================================

def perform_checks(
    document_type,
    text,
    fields
):

    checks = []


    if document_type == "unknown":

        checks.append({

            "status": "WARN",

            "message":
                "Automatic document type could not be confidently detected."

        })

        return checks


    rule = DOCUMENT_RULES[
        document_type
    ]


    normalized = normalize_text(
        text
    )


    # --------------------------------------------------------
    # KEYWORD CHECKS
    # --------------------------------------------------------

    for keyword in rule["keywords"]:

        if keyword.lower() in normalized:

            checks.append({

                "status": "PASS",

                "message":
                    f"Format indicator detected: {keyword}"

            })

        else:

            checks.append({

                "status": "WARN",

                "message":
                    f"Expected format indicator not detected: {keyword}"

            })


    # --------------------------------------------------------
    # FIELD CHECKS
    # --------------------------------------------------------

    for field in rule["fields"]:

        value = fields.get(
            field
        )


        if value:

            checks.append({

                "status": "PASS",

                "message":
                    f"Field detected: {field.replace('_', ' ').title()}"

            })

        else:

            checks.append({

                "status": "WARN",

                "message":
                    f"Field not detected: {field.replace('_', ' ').title()}"

            })


    # --------------------------------------------------------
    # MRZ
    # --------------------------------------------------------

    if rule["mrz"]:

        if detect_mrz(text):

            checks.append({

                "status": "PASS",

                "message":
                    "Passport MRZ-like structure detected."

            })

        else:

            checks.append({

                "status": "WARN",

                "message":
                    "Passport MRZ-like structure not detected."

            })


    return checks


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_quality(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


    if sharpness >= 500:

        score = 100

    elif sharpness >= 300:

        score = 90

    elif sharpness >= 180:

        score = 80

    elif sharpness >= 100:

        score = 70

    elif sharpness >= 50:

        score = 55

    else:

        score = 35


    return score


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(
    document_type,
    text,
    fields
):

    if document_type == "unknown":

        return 20


    rule = DOCUMENT_RULES[
        document_type
    ]


    normalized = normalize_text(text)


    keyword_hits = sum(
        1
        for keyword in rule["keywords"]
        if keyword.lower() in normalized
    )


    keyword_score = (
        keyword_hits /
        max(len(rule["keywords"]), 1)
    ) * 100


    field_hits = sum(
        1
        for field in rule["fields"]
        if fields.get(field)
    )


    field_score = (
        field_hits /
        max(len(rule["fields"]), 1)
    ) * 100


    if rule["mrz"]:

        mrz_score = (
            100
            if detect_mrz(text)
            else 0
        )

        final = (
            keyword_score * 0.35 +
            field_score * 0.45 +
            mrz_score * 0.20
        )

    else:

        final = (
            keyword_score * 0.40 +
            field_score * 0.60
        )


    return round(
        max(0, min(100, final))
    )


# ============================================================
# FIELD CONSISTENCY
# ============================================================

def calculate_consistency(
    document_type,
    fields
):

    if document_type == "unknown":

        return 20


    required = DOCUMENT_RULES[
        document_type
    ]["fields"]


    present = sum(
        1
        for field in required
        if fields.get(field)
    )


    score = (
        present /
        max(len(required), 1)
    ) * 100


    return round(
        score
    )


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(
    quality,
    structure,
    consistency
):

    screening_score = (

        quality * 0.25 +

        structure * 0.50 +

        consistency * 0.25

    )


    risk = 100 - screening_score


    risk = round(
        max(0, min(100, risk))
    )


    if risk >= 60:

        status = "HIGH RISK REVIEW"

    elif risk >= 30:

        status = "MANUAL REVIEW"

    else:

        status = "LOW RISK SIGNAL"


    return risk, status


# ============================================================
# MISMATCH FILTER
# ============================================================

def build_mismatch_summary(
    document_type,
    checks,
    fields
):

    mismatches = []


    for check in checks:

        if check["status"] != "PASS":

            mismatches.append(
                check["message"]
            )


    if document_type == "unknown":

        mismatches.insert(
            0,
            "Document format could not be confidently classified."
        )


    if not mismatches:

        mismatches.append(
            "No major format indicators were missing."
        )


    return mismatches[:10]


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Invalid file."
        )


    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only image documents are supported."
        )


    # --------------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------------

    extension = os.path.splitext(
        file.filename or ""
    )[1].lower()


    if extension not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ]:

        extension = ".jpg"


    filename = (
        str(uuid.uuid4())
        + extension
    )


    filepath = os.path.join(
        UPLOAD_DIR,
        filename
    )


    contents = await file.read()


    with open(
        filepath,
        "wb"
    ) as f:

        f.write(contents)


    # --------------------------------------------------------
    # DECODE IMAGE
    # --------------------------------------------------------

    image_array = np.frombuffer(
        contents,
        np.uint8
    )


    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Could not decode document image."
        )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    raw_text = extract_text(
        image
    )


    # --------------------------------------------------------
    # AUTOMATIC DOCUMENT DETECTION
    # --------------------------------------------------------

    (
        document_type,
        confidence,
        detection_scores
    ) = detect_document_type(
        raw_text
    )


    # --------------------------------------------------------
    # FIELD EXTRACTION
    # --------------------------------------------------------

    fields = extract_fields(
        raw_text
    )


    # --------------------------------------------------------
    # CHECKS
    # --------------------------------------------------------

    checks = perform_checks(
        document_type,
        raw_text,
        fields
    )


    # --------------------------------------------------------
    # SCORING
    # --------------------------------------------------------

    quality = calculate_quality(
        image
    )


    structure = calculate_structure_score(
        document_type,
        raw_text,
        fields
    )


    consistency = calculate_consistency(
        document_type,
        fields
    )


    risk, status = calculate_risk(
        quality,
        structure,
        consistency
    )


    format_match = (
        100 - risk
    )


    # --------------------------------------------------------
    # MISMATCHES
    # --------------------------------------------------------

    mismatches = build_mismatch_summary(
        document_type,
        checks,
        fields
    )


    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_id = (
        "BS-"
        + datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )
        + "-"
        + uuid.uuid4().hex[:5].upper()
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "success": True,

        "reportId": report_id,

        "documentType":
            document_type.replace(
                "_",
                " "
            ).upper(),

        "documentTypeConfidence":
            confidence,

        "detectionScores":
            detection_scores,

        "formatMatch":
            format_match,

        "risk":
            risk,

        "status":
            status,

        "quality":
            quality,

        "structure":
            structure,

        "consistency":
            consistency,

        "fields":
            fields,

        "checks":
            checks,

        "mismatches":
            mismatches,

        "rawText":
            raw_text,

        "governmentVerification": {

            "status":
                "NOT CONNECTED",

            "message":
                "No government verification API is connected in this prototype."

        },

        "timestamp":
            datetime.now().isoformat()

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return FileResponse(FRONTEND_FILE)


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "service":
            "BLACK SECURITY"

    }