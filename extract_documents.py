"""
============================================================
SCRIPT 1 : EXTRACTION TEXTE + TABLEAUX + OCR
PROJET TABAC TUNISIE
============================================================

Extraction :
- PPTX : texte natif + tableaux + texte dans les images
         (y compris images imbriquees dans des groupes)

- PDF  : texte natif (pdfplumber) + tableaux + images
         embarquees (PyMuPDF + OCR) + fallback OCR pleine
         page pour les PDF scannes

Objectif :
Produire des fichiers .txt propres et structures avant
chunking, retrieval et construction du Knowledge Graph.
============================================================
"""

from pathlib import Path
from difflib import SequenceMatcher
import subprocess
import tempfile

import io
import os
import re

import pdfplumber
import fitz  # PyMuPDF

import pytesseract
from pytesseract import Output
from PIL import Image

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

# Chemins configurables via .env (valeurs par defaut = installation Windows
# habituelle de ce poste) -- utile si Tesseract/LibreOffice sont installes
# ailleurs ou sur une autre plateforme.
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
SOFFICE_PATH = os.getenv(
    "SOFFICE_PATH", r"C:\Program Files\LibreOffice\program\soffice.exe"
)
# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "./documents"
OUTPUT_FOLDER = "./textes"

# Langues OCR
OCR_LANG = "fra+eng"

# Images trop petites = probablement logo / icone
MIN_IMAGE_SIZE = 80

# Seuil de similarite pour considerer deux textes
# comme probablement identiques
SIMILARITY_THRESHOLD = 0.90

# Nombre minimum de mots pour considerer qu'une page
# contient du texte natif exploitable
MIN_MEANINGFUL_WORDS = 5

# Resolution du fallback OCR pleine page
PDF_OCR_SCALE = 3


# ============================================================
# NETTOYAGE / NORMALISATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalise un texte pour faciliter la comparaison entre
    texte natif et texte OCR.
    """

    if not text:
        return ""

    text = text.lower()

    # Normalisation des espaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def has_meaningful_text(text: str) -> bool:
    """
    Determine si un texte contient suffisamment de contenu
    pour etre considere comme du vrai texte exploitable.

    Cela evite de considerer comme suffisante une page contenant
    seulement quelques caracteres parasites.
    """

    normalized = normalize_text(text)

    if not normalized:
        return False

    words = normalized.split()

    return len(words) >= MIN_MEANINGFUL_WORDS


# ============================================================
# DETECTION DE DOUBLONS / SIMILARITE
# ============================================================

def is_similar_text(text1: str, text2: str) -> bool:
    """
    Compare deux textes pour detecter un doublon probable.

    Strategie :

    1. Normalisation
    2. Inclusion directe
    3. Comparaison globale
    4. Si un texte est beaucoup plus court que l'autre :
       comparaison du texte court avec des fenetres de taille
       similaire dans le texte long.

    Cette derniere etape est importante pour comparer par exemple :
    - OCR d'une image contenant une phrase
    - texte natif d'une page entiere beaucoup plus longue
    """

    t1 = normalize_text(text1)
    t2 = normalize_text(text2)

    if not t1 or not t2:
        return False

    # --------------------------------------------------------
    # 1. Inclusion directe
    # --------------------------------------------------------

    if t1 in t2 or t2 in t1:
        return True

    # --------------------------------------------------------
    # 2. Comparaison globale
    # --------------------------------------------------------

    ratio = SequenceMatcher(None, t1, t2).ratio()

    if ratio >= SIMILARITY_THRESHOLD:
        return True

    # --------------------------------------------------------
    # 3. Comparaison par fenetres
    # --------------------------------------------------------
    # On identifie le texte court et le texte long.

    if len(t1) < len(t2):
        short_text = t1
        long_text = t2
    else:
        short_text = t2
        long_text = t1

    # Si les textes ont une taille similaire, inutile
    # de parcourir des fenetres.
    if len(short_text) >= len(long_text):
        return False

    # Fenetre de la MEME taille que le texte court.
    # Une fenetre 2x plus grande diluerait artificiellement
    # le score SequenceMatcher.
    window_size = len(short_text)

    # Avancement de 25 % de la fenetre.
    # Minimum 10 caracteres pour eviter trop de calculs.
    step = max(window_size // 4, 10)

    # On parcourt aussi la derniere position possible.
    max_start = len(long_text) - window_size

    starts = list(range(0, max_start + 1, step))

    if starts and starts[-1] != max_start:
        starts.append(max_start)

    for start in starts:

        segment = long_text[start:start + window_size]

        if not segment:
            continue

        ratio = SequenceMatcher(
            None,
            short_text,
            segment
        ).ratio()

        if ratio >= SIMILARITY_THRESHOLD:
            return True

    return False


# ============================================================
# OCR
# ============================================================

def ocr_image(image) -> str:
    """
    Applique l'OCR sur :

    - bytes
    - PIL.Image.Image

    Retourne le texte detecte.
    Retourne une chaine vide si aucun texte utile n'est trouve.
    """

    try:

        # ----------------------------------------------------
        # Conversion bytes -> PIL Image
        # ----------------------------------------------------

        if isinstance(image, bytes):

            img = Image.open(
                io.BytesIO(image)
            )

        else:

            img = image

        # ----------------------------------------------------
        # Verification de la taille
        # ----------------------------------------------------

        if (
            img.width < MIN_IMAGE_SIZE
            or
            img.height < MIN_IMAGE_SIZE
        ):
            return ""

        # ----------------------------------------------------
        # Conversion RGB
        # Important pour :
        # - CMYK
        # - RGBA
        # - images avec alpha
        # ----------------------------------------------------

        if img.mode != "RGB":

            img = img.convert("RGB")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = pytesseract.image_to_string(
            img,
            lang=OCR_LANG
        )

        return text.strip()

    except Exception as e:

        print(f"   ⚠️ OCR echoue : {e}")

        return ""


def ocr_image_layout_aware(image) -> str:
    """
    OCR avec reconstruction de la mise en page.

    Separe CHAQUE MOT en colonne gauche / droite selon sa
    position horizontale individuelle (pas la ligne entiere),
    puis reconstruit les lignes a l'interieur de chaque colonne
    par regroupement vertical.
    """

    if isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        img = image

    if img.mode != "RGB":
        img = img.convert("RGB")

    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, output_type=Output.DICT
    )

    mid_x = img.width / 2

    left_words = []
    right_words = []

    for i in range(len(data["text"])):

        word = data["text"][i].strip()

        if not word:
            continue

        left = data["left"][i]
        top = data["top"][i]
        width = data["width"][i]

        # Centre du mot, plus fiable que le bord gauche
        center_x = left + width / 2

        entry = {"text": word, "left": left, "top": top}

        if center_x < mid_x:
            left_words.append(entry)
        else:
            right_words.append(entry)

    def reconstruct_column(words, line_tolerance=15):
        """
        Regroupe les mots d'une colonne en lignes, selon
        leur proximite verticale (top), puis trie chaque
        ligne par position horizontale.
        """

        if not words:
            return ""

        words_sorted = sorted(words, key=lambda w: w["top"])

        rows = []
        current_row = [words_sorted[0]]
        current_top = words_sorted[0]["top"]

        for w in words_sorted[1:]:

            if abs(w["top"] - current_top) <= line_tolerance:
                current_row.append(w)
            else:
                rows.append(current_row)
                current_row = [w]
                current_top = w["top"]

        rows.append(current_row)

        lines = []
        for row in rows:
            row_sorted = sorted(row, key=lambda w: w["left"])
            lines.append(" ".join(w["text"] for w in row_sorted))

        return "\n".join(lines)

    left_text = reconstruct_column(left_words)
    right_text = reconstruct_column(right_words)

    return (left_text + "\n---\n" + right_text).strip()
def convert_vector_and_ocr(image_bytes, ext) -> str:
    """
    Convertit une image WMF/EMF en PDF (vectoriel, sans perte)
    via LibreOffice, puis rasterise ce PDF a haute resolution
    avec PyMuPDF avant l'OCR.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        input_path = Path(tmp_dir) / f"image.{ext}"

        with open(input_path, "wb") as f:
            f.write(image_bytes)

        # ----------------------------------------------------
        # CONVERSION WMF/EMF -> PDF (vectoriel)
        # ----------------------------------------------------

        try:
            subprocess.run(
                [SOFFICE_PATH, "--headless", "--convert-to", "pdf",
                 "--outdir", tmp_dir, str(input_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30
            )
        except Exception as e:
            print(f"   ⚠️ Conversion {ext} echouee : {e}")
            return ""

        pdf_path = Path(tmp_dir) / "image.pdf"

        if not pdf_path.exists():
            return ""

        # ----------------------------------------------------
        # RASTERISATION HAUTE RESOLUTION AVEC PYMUPDF
        # ----------------------------------------------------

        try:
            doc = fitz.open(str(pdf_path))
            page = doc[0]

            # Meme logique que PDF_OCR_SCALE, mais plus fort
            # car ces images sont petites a la base
            pix = page.get_pixmap(
                matrix=fitz.Matrix(6, 6)  # ~432 DPI
            )

            img = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            doc.close()

            return ocr_image_layout_aware(img)

        except Exception as e:
            print(f"   ⚠️ Rasterisation {ext} echouee : {e}")
            return ""

# ============================================================
# TABLEAUX PPTX
# ============================================================

def extract_table_as_md(table):
    """
    Convertit un tableau PowerPoint en Markdown.
    """

    lines = []

    for i, row in enumerate(table.rows):

        cells = [
            cell.text.strip().replace("\n", " ")
            for cell in row.cells
        ]

        lines.append(
            "| " + " | ".join(cells) + " |"
        )

        # Premiere ligne = en-tete
        if i == 0:

            lines.append(
                "| "
                + " | ".join(
                    ["---"] * len(cells)
                )
                + " |"
            )

    return "\n".join(lines)


# ============================================================
# EXTRACTION IMAGES PPTX
# ============================================================

def extract_images_from_shapes(shapes) -> list:
    """
    Parcourt toutes les formes d'une diapositive.

    Gere :
    - images classiques
    - images imbriquees dans des groupes
    - images vectorielles (WMF/EMF) via conversion

    Retourne les textes OCR detectes.
    """

    ocr_results = []

    for shape in shapes:

        # ----------------------------------------------------
        # IMAGE CLASSIQUE
        # ----------------------------------------------------

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:

            try:

                image_bytes = shape.image.blob
                image_ext = shape.image.ext

                # ------------------------------------------------
                # IMAGE VECTORIELLE (WMF/EMF)
                # ------------------------------------------------

                if image_ext in ("wmf", "emf"):

                    text = convert_vector_and_ocr(
                        image_bytes,
                        image_ext
                    )

                # ------------------------------------------------
                # IMAGE BITMAP CLASSIQUE
                # ------------------------------------------------

                else:

                    text = ocr_image(
                        image_bytes
                    )

                if text:

                    ocr_results.append(
                        text
                    )

            except Exception as e:

                print(
                    f"   ⚠️ Impossible de lire "
                    f"une image PPTX : {e}"
                )

        # ----------------------------------------------------
        # GROUPE DE FORMES
        # ----------------------------------------------------

        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:

            nested_results = (
                extract_images_from_shapes(
                    shape.shapes
                )
            )

            ocr_results.extend(
                nested_results
            )

    return ocr_results
# ============================================================
# EXTRACTION PPTX
# ============================================================

def extract_text_from_pptx(
    pptx_path,
    output_path
):
    """
    Extrait :

    1. Texte natif
    2. Tableaux
    3. Texte contenu dans les images via OCR
    """

    prs = Presentation(
        pptx_path
    )

    text_content = []

    # ========================================================
    # PARCOURS DES DIAPOSITIVES
    # ========================================================

    for i, slide in enumerate(
        prs.slides,
        1
    ):

        slide_text = []

        # ----------------------------------------------------
        # 1. TEXTE NATIF + TABLEAUX
        # ----------------------------------------------------

        for shape in slide.shapes:

            if shape.has_table:

                slide_text.append(
                    "=== TABLEAU ==="
                )

                slide_text.append(
                    extract_table_as_md(
                        shape.table
                    )
                )

                slide_text.append(
                    "=== FIN TABLEAU ==="
                )

            elif (
                hasattr(shape, "text")
                and shape.text.strip()
            ):

                slide_text.append(
                    shape.text.strip()
                )

        # Texte natif avant ajout de l'OCR
        native_text = "\n".join(
            slide_text
        )

        # ----------------------------------------------------
        # 2. OCR DES IMAGES
        # ----------------------------------------------------

        ocr_texts = (
            extract_images_from_shapes(
                slide.shapes
            )
        )

        for ocr_text in ocr_texts:

            # Si le texte de l'image semble deja
            # present dans le texte natif, on ne l'ajoute pas.
            if is_similar_text(
                ocr_text,
                native_text
            ):

                continue

            slide_text.append(
                "=== TEXTE OCR (IMAGE) ==="
            )

            slide_text.append(
                ocr_text
            )

            slide_text.append(
                "=== FIN OCR ==="
            )

        # ----------------------------------------------------
        # 3. SAUVEGARDE DIAPOSITIVE
        # ----------------------------------------------------

        if slide_text:

            text_content.append(
                f"=== DIAPOSITIVE {i} ==="
            )

            text_content.append(
                "\n".join(
                    slide_text
                )
            )

            text_content.append(
                ""
            )

    # ========================================================
    # ECRITURE FICHIER
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                text_content
            )
        )

    print(
        f"✅ PPTX : "
        f"{os.path.basename(pptx_path)}"
    )


# ============================================================
# EXTRACTION PDF
# ============================================================

def extract_text_from_pdf(
    pdf_path,
    output_path
):
    """
    Extrait un PDF en combinant :

    1. Texte natif avec pdfplumber
    2. Tableaux avec pdfplumber
    3. OCR des images embarquees avec PyMuPDF
    4. Fallback OCR pleine page pour les PDF scannes
    """

    text_content = []

    # On garde separement le texte natif de chaque page.
    #
    # Important :
    # C'est exactement ce texte qui est ecrit dans le .txt,
    # donc c'est celui qu'on utilise ensuite pour comparer
    # l'OCR et eviter les doublons.
    native_texts = []

    # ========================================================
    # 1. TEXTE NATIF + TABLEAUX
    # ========================================================

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        for i, page in enumerate(
            pdf.pages,
            1
        ):

            # ------------------------------------------------
            # TEXTE NATIF
            # ------------------------------------------------

            text = (
                page.extract_text()
                or ""
            )

            native_texts.append(
                text
            )

            # ------------------------------------------------
            # TABLEAUX
            # ------------------------------------------------

            tables = (
                page.extract_tables()
            )

            page_parts = [
                f"=== PAGE {i} ==="
            ]

            if text.strip():

                page_parts.append(
                    text
                )

            # ------------------------------------------------
            # AJOUT TABLEAUX
            # ------------------------------------------------

            for j, table in enumerate(
                tables,
                1
            ):

                page_parts.append(
                    f"=== TABLEAU {j} ==="
                )

                for row in table:

                    cells = [
                        str(cell).strip()
                        if cell
                        else ""
                        for cell in row
                    ]

                    page_parts.append(
                        "| "
                        + " | ".join(cells)
                        + " |"
                    )

                page_parts.append(
                    "=== FIN TABLEAU ==="
                )

            text_content.append(
                "\n".join(
                    page_parts
                )
            )

    # ========================================================
    # 2. OCR AVEC PYMUPDF
    # ========================================================

    doc = fitz.open(
        pdf_path
    )

    for page_index in range(
        len(doc)
    ):

        page = doc[
            page_index
        ]

        native_text = native_texts[
            page_index
        ]

        # Texte OCR qui sera ajoute au fichier final
        ocr_parts = []

        # ----------------------------------------------------
        # IMPORTANT :
        #
        # found_ocr_text indique si un OCR a detecte
        # du texte, meme si ce texte est ensuite ignore
        # parce qu'il est deja present dans le texte natif.
        #
        # Cela empeche de lancer inutilement le fallback
        # OCR pleine page.
        # ----------------------------------------------------

        found_ocr_text = False

        # ====================================================
        # 2A. OCR DES IMAGES EMBARQUEES
        # ====================================================

        images = page.get_images(
            full=True
        )

        seen_xrefs = set()

        for img_index, img in enumerate(
            images,
            1
        ):

            xref = img[0]

            # Evite de traiter plusieurs fois
            # la meme image.
            if xref in seen_xrefs:

                continue

            seen_xrefs.add(
                xref
            )

            try:

                base_image = (
                    doc.extract_image(
                        xref
                    )
                )

                image_bytes = (
                    base_image[
                        "image"
                    ]
                )

                ocr_text = ocr_image(
                    image_bytes
                )

                # Aucun texte detecte
                if not ocr_text:

                    continue

                # Du texte a ete trouve
                found_ocr_text = True

                # ------------------------------------------------
                # DEJA PRESENT DANS LE TEXTE NATIF ?
                # ------------------------------------------------

                if is_similar_text(
                    ocr_text,
                    native_text
                ):

                    continue

                # ------------------------------------------------
                # AJOUT DU TEXTE OCR
                # ------------------------------------------------

                ocr_parts.append(
                    f"=== TEXTE OCR "
                    f"(IMAGE {img_index}) ==="
                )

                ocr_parts.append(
                    ocr_text
                )

                ocr_parts.append(
                    "=== FIN OCR ==="
                )

            except Exception as e:

                print(
                    f"   ⚠️ Erreur image PDF "
                    f"page {page_index + 1} : {e}"
                )

        # ====================================================
        # 2B. FALLBACK OCR PLEINE PAGE
        # ====================================================
        #
        # On lance le fallback seulement si :
        #
        # - le texte natif n'est pas significatif
        # - aucune image embarquee n'a fourni de texte
        #
        # found_ocr_text est utilise et NON ocr_parts.
        #
        # Une image peut contenir du texte, mais etre ignoree
        # car ce texte est deja present dans le natif.
        # Dans ce cas, inutile de relancer un OCR pleine page.
        # ====================================================

        if (
            not has_meaningful_text(
                native_text
            )
            and not found_ocr_text
        ):

            try:

                print(
                    f"   🔍 OCR pleine page "
                    f"{page_index + 1}..."
                )

                # ------------------------------------------------
                # 3x = environ 216 DPI
                # ------------------------------------------------

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        PDF_OCR_SCALE,
                        PDF_OCR_SCALE
                    )
                )

                img = Image.open(
                    io.BytesIO(
                        pix.tobytes(
                            "png"
                        )
                    )
                )

                page_ocr = ocr_image(
                    img
                )

                if page_ocr:

                    ocr_parts.append(
                        "=== TEXTE OCR "
                        "(PAGE COMPLETE) ==="
                    )

                    ocr_parts.append(
                        page_ocr
                    )

                    ocr_parts.append(
                        "=== FIN OCR ==="
                    )

            except Exception as e:

                print(
                    f"   ⚠️ OCR pleine page "
                    f"echoue page "
                    f"{page_index + 1} : {e}"
                )

        # ====================================================
        # 3. AJOUT OCR A LA PAGE
        # ====================================================

        if ocr_parts:

            text_content[
                page_index
            ] += (
                "\n"
                + "\n".join(
                    ocr_parts
                )
            )

    # Fermeture PDF
    doc.close()

    # ========================================================
    # 4. SAUVEGARDE
    # ========================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n\n".join(
                text_content
            )
        )

    print(
        f"✅ PDF  : "
        f"{os.path.basename(pdf_path)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Creation du dossier de sortie
    Path(
        OUTPUT_FOLDER
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PPTX
    # ========================================================

    pptx_files = sorted(
        Path(
            INPUT_FOLDER
        ).glob(
            "*.pptx"
        )
    )

    for pptx_file in pptx_files:

        output_path = os.path.join(
            OUTPUT_FOLDER,
            pptx_file.stem + ".txt"
        )

        extract_text_from_pptx(
            str(pptx_file),
            output_path
        )

    # ========================================================
    # PDF
    # ========================================================

    pdf_files = sorted(
        Path(
            INPUT_FOLDER
        ).glob(
            "*.pdf"
        )
    )

    for pdf_file in pdf_files:

        output_path = os.path.join(
            OUTPUT_FOLDER,
            pdf_file.stem + ".txt"
        )

        extract_text_from_pdf(
            str(pdf_file),
            output_path
        )

    # ========================================================
    # FIN
    # ========================================================

    total = (
        len(pptx_files)
        + len(pdf_files)
    )

    print(
        f"\n🎉 {total} document(s) "
        f"extrait(s) dans "
        f"./{OUTPUT_FOLDER}/"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()