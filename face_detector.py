"""
Created on 04.05.2022
Modified on 01.03.2025
by @author: Paideia
send the the comparison and improvements of algorithms to the Profesor´s mail
manuel.quispe.t@uni.edu.pe
2025-II Distributed Computer Vision
"""

# ----------------- Forzar TensorFlow/MTCNN con CPU -------------------------
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # desactiva GPU en TF/MTCNN si existiera
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos logs de TF

# ------------------------------- IMPORTS ----------------------------------------
from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import time
from collections import defaultdict

# ============================ CONFIGURACIÓN ============================
IMG_DIR = 'my_faces'
HAAR_PATH = 'face_detection/models/haarcascade_frontalface2.xml'
CAFFE_MODEL = r"models/res10_300x300_ssd_iter_140000.caffemodel"
CAFFE_PROTO  = r"models/deploy.prototxt.txt"

# -------- Upscale por detector (para caras pequeñas) --------
USE_UPSCALE     = dict(mtcnn=True,  dlib=True,  dnn=True,  haar=True)
UPSCALE_FACTOR  = dict(mtcnn=1.5,    dlib=1.5,   dnn=1.5,   haar=1.5)

# --------------------------- Grayscale ------------------------------
USE_GRAYSCALE = dict(mtcnn=False, dlib=False,  dnn=False, haar=False)
USE_ADIFF     = dict(mtcnn=False, dlib=False, dnn=False, haar=False)

# Mejoras fotométricas
USE_CLAHE   = dict(mtcnn=False, dlib=False,  dnn=False, haar=False)
USE_GAMMA   = dict(mtcnn=False, dlib=False,  dnn=False, haar=False)
USE_BRIGHT  = dict(mtcnn=False, dlib=False,  dnn=False, haar=False)

GAMMA_VALUE  = 1.1
BRIGHT_VALUE = 15

# Slicing
USE_SLICING = dict(mtcnn=False, dlib=False, dnn=False, haar=False)
SLICE_CFG  = dict(tile=150, overlap=0.2)
SLICE_HAAR = dict(tile=150, overlap=0.7)

# NMS razonable
USE_NMS     = dict(mtcnn=False,  dlib=True,  dnn=True,  haar=True)
NMS_IOU     = dict(mtcnn=0.50,  dlib=0.50,  dnn=0.50,  haar=0.30)

# Umbrales (los que sí existen como confianza directa)
CONF = dict(mtcnn=0.60, dnn=0.60)

# ===================== UMBRALES / PARÁMETROS HAAR y HOG =====================
# HAAR: "umbral" indirecto
HAAR_CFG = dict(
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(24, 24)
)

# DLIB HOG: umbral real
HOG_CFG = dict(
    upsample=2,
    adjust_threshold=0.0,
    score_thr=None  # ejemplo: 0.2
)

# Logs
LOG_TIMES = True
SAVE_LOGS_CSV = False
LOGS_CSV_PATH = 'timings_faces.csv'

SHOW = True
SAVE = False
SAVE_DIR = 'faces_out'
# ===================================================================================================

# ====================== DIFUSIÓN ANISOTRÓPICA (Perona–Malik) ====================
def anisotropic_diffusion_gray(img_gray, niter=8, k=20.0, lam=0.25, option=1):
    I = img_gray.astype(np.float32)
    for _ in range(int(niter)):
        north = np.roll(I, -1, axis=0) - I
        south = np.roll(I,  1, axis=0) - I
        east  = np.roll(I,  1, axis=1) - I
        west  = np.roll(I, -1, axis=1) - I

        if option == 1:
            cN = np.exp(-(north/k)**2); cS = np.exp(-(south/k)**2)
            cE = np.exp(-(east/k)**2);  cW = np.exp(-(west/k)**2)
        else:
            cN = 1.0 / (1.0 + (north/k)**2); cS = 1.0 / (1.0 + (south/k)**2)
            cE = 1.0 / (1.0 + (east/k)**2);  cW = 1.0 / (1.0 + (west/k)**2)

        I += lam * (cN*north + cS*south + cE*east + cW*west)

    return np.clip(I, 0, 255).astype(np.uint8)

def anisotropic_diffusion_color(img_bgr, **kwargs):
    b, g, r = cv2.split(img_bgr)
    b = anisotropic_diffusion_gray(b, **kwargs)
    g = anisotropic_diffusion_gray(g, **kwargs)
    r = anisotropic_diffusion_gray(r, **kwargs)
    return cv2.merge([b, g, r])

# --------------------------- UPSCALE ---------------------------------
def upscale_image(img_bgr, factor=1.0, interp=cv2.INTER_CUBIC):
    """
    Upscaling manteniendo aspect ratio.
    - factor > 1.0 para agrandar; <= 1.0 no cambia.
    Devuelve: (img_upscaled, factor_aplicado)
    """
    if factor is None or factor <= 1.0:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    new_w = int(round(w * factor))
    new_h = int(round(h * factor))
    up = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    return up, float(factor)

# ============================ FUNCIONES DE MEJORA ===============================
def apply_clahe_bgr(img, clip=2.0, grid=8, blend=0.6):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    l_out = cv2.addWeighted(l, 1.0 - blend, l2, blend, 0) if 0.0 <= blend < 1.0 else l2
    return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)

def apply_clahe_gray(gray, clip=2.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return clahe.apply(gray)

def gamma_correction(img, gamma=1.0):
    if abs(gamma - 1.0) < 1e-3:
        return img
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv * 255).astype("uint8")
    return cv2.LUT(img, table)

def gamma_correction_gray(gray, gamma=1.0):
    if abs(gamma - 1.0) < 1e-3:
        return gray
    inv = 1.0 / max(gamma, 1e-6)
    table = (np.linspace(0, 1, 256) ** inv * 255).astype("uint8")
    return cv2.LUT(gray, table)

def bright_correction(img, bright=0):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=float(bright))

def bright_correction_gray(gray, bright=0):
    return cv2.convertScaleAbs(gray, alpha=1.0, beta=float(bright))

# --------------------------- HELPERS DE CAJAS -----------------------------------
def descale_boxes(boxes, scale):
    if scale is None or abs(scale - 1.0) < 1e-6:
        return boxes
    inv = 1.0 / scale
    out = []
    for (x1, y1, x2, y2) in boxes:
        out.append([int(round(x1 * inv)), int(round(y1 * inv)),
                    int(round(x2 * inv)), int(round(y2 * inv))])
    return out

def clamp_boxes(boxes, w, h):
    clamped = []
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
        if x2 > x1 and y2 > y1:
            clamped.append([x1, y1, x2, y2])
    return clamped

# ------------------------------- NMS -------------------------------------------
def _iou(b, Bs):
    x1 = np.maximum(b[0], Bs[:, 0]); y1 = np.maximum(b[1], Bs[:, 1])
    x2 = np.minimum(b[2], Bs[:, 2]); y2 = np.minimum(b[3], Bs[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    ab = (b[2]-b[0]) * (b[3]-b[1])
    As = (Bs[:, 2]-Bs[:, 0]) * (Bs[:, 3]-Bs[:, 1])
    return inter / np.maximum(ab + As - inter, 1e-6)

def nms(boxes, scores, iou_thr=0.45):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = _iou(boxes[i], boxes[rest])
        idxs = rest[ious < iou_thr]
    return [boxes[k].astype(int).tolist() for k in keep], [float(scores[k]) for k in keep]

# ------------------------- Slicing -----------------------------------
def slice_and_detect(img_bgr, detect_fn, tile=640, overlap=0.2, **kwargs):
    H, W = img_bgr.shape[:2]
    stride = max(1, int(tile * (1.0 - overlap)))
    boxes_all, scores_all = [], []
    for y0 in range(0, max(1, H - tile + 1), stride):
        for x0 in range(0, max(1, W - tile + 1), stride):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            tile_img = img_bgr[y0:y1, x0:x1]
            b, s = detect_fn(tile_img, **kwargs)
            for bb, ss in zip(b, s):
                boxes_all.append([bb[0]+x0, bb[1]+y0, bb[2]+x0, bb[3]+y0])
                scores_all.append(ss)
    return boxes_all, scores_all

# ============================= DETECTORES (unificados) ==========================
def detect_mtcnn(img_bgr, mtcnn, conf_th=0.5):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mtcnn.detect_faces(rgb)
    boxes, scores = [], []
    for r in res:
        c = float(r.get('confidence', 1.0))
        if c >= conf_th:
            x, y, w, h = r['box']
            x, y = max(0, x), max(0, y)
            boxes.append([x, y, x+w, y+h])
            scores.append(c)
    return boxes, scores

def detect_dlib_hog(img_bgr, hog, upsample=2, adjust_threshold=0.0, score_thr=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    rects, scores, _ = hog.run(gray, int(upsample), float(adjust_threshold))

    if score_thr is not None:
        rects_f, scores_f = [], []
        for r, s in zip(rects, scores):
            if float(s) >= float(score_thr):
                rects_f.append(r)
                scores_f.append(s)
        rects, scores = rects_f, scores_f

    boxes = [[r.left(), r.top(), r.right(), r.bottom()] for r in rects]
    scores = [float(s) for s in scores]
    return boxes, scores

def detect_dnn(img_bgr, net, conf_th=0.5):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)),
        1.0, (300, 300),
        (104, 117, 123)
    )
    net.setInput(blob)
    out = net.forward()

    boxes, scores = [], []
    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        if score >= conf_th:
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    return boxes, scores

def detect_haar(img_bgr, cascade, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=float(scaleFactor),
        minNeighbors=int(minNeighbors),
        minSize=tuple(minSize)
    )
    boxes = [[x, y, x+w, y+h] for (x, y, w, h) in faces]
    scores = [1.0] * len(boxes)
    return boxes, scores

# ============================== INICIALIZACIÓN =================================
detector1 = MTCNN()                            # MTCNN
detector2 = dlib.get_frontal_face_detector()   # DLIB-HOG
net = cv2.dnn.readNetFromCaffe(CAFFE_PROTO, CAFFE_MODEL)  # DNN-SSD Caffe

# Fuerza DNN a CPU (evita errores de CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("[DNN] Backend=OPENCV  Target=CPU")

classifier2 = cv2.CascadeClassifier(HAAR_PATH) # HAAR

images = os.listdir(IMG_DIR)
print("\n\n\nTEXTO DE LA DIRECCIÓN:", images)

if SAVE and not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# Logs agregados
logs_per_detector = defaultdict(lambda: {
    'pre_ms': 0.0, 'det_ms': 0.0, 'post_ms': 0.0, 'total_ms': 0.0, 'count': 0
})
logs_rows = []

# ============================ PREPROCESS ============================
def preprocess(img, det_name):
    """
    Retorna: (out_bgr, scale)
    - Si USE_GRAYSCALE[det_name] es True: opera en gris y vuelve a BGR.
    - Si es False: opera en color.
    - Aplica UPSCALE justo antes del ADIFF, luego el resto (CLAHE, gamma, bright).
    """
    out = img
    scale = 1.0

    # --- UPSCALE (antes del ADIFF) ---
    if USE_UPSCALE.get(det_name, False):
        factor = float(UPSCALE_FACTOR.get(det_name, 1.0))
        if factor > 1.0:
            out, scale = upscale_image(out, factor=factor, interp=cv2.INTER_CUBIC)

    if USE_GRAYSCALE.get(det_name, False):
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        if USE_ADIFF.get(det_name, False):
            gray = anisotropic_diffusion_gray(gray, niter=8, k=20.0, lam=0.25, option=1)

        if USE_CLAHE.get(det_name, False):
            gray = apply_clahe_gray(gray, clip=2.0, grid=8)

        if USE_GAMMA.get(det_name, False):
            gray = gamma_correction_gray(gray, GAMMA_VALUE)

        if USE_BRIGHT.get(det_name, False):
            gray = bright_correction_gray(gray, BRIGHT_VALUE)

        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        if USE_ADIFF.get(det_name, False):
            out = anisotropic_diffusion_color(out, niter=8, k=20.0, lam=0.25, option=1)

        if USE_CLAHE.get(det_name, False):
            out = apply_clahe_bgr(out, clip=2.0, grid=8, blend=0.6)

        if USE_GAMMA.get(det_name, False):
            out = gamma_correction(out, GAMMA_VALUE)

        if USE_BRIGHT.get(det_name, False):
            out = bright_correction(out, BRIGHT_VALUE)

    return out, scale

# =================== Utilidades de visualización (montaje 2x2) ==================
def _fit_letterbox(img, size=(480, 360), bg=(0, 0, 0)):
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(
        img, (nw, nh),
        interpolation=cv2.INTER_AREA if (nw < w or nh < h) else cv2.INTER_CUBIC
    )
    canvas = np.full((th, tw, 3), bg, dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _add_title_bar(img, title, bar_h=36, bg=(32, 32, 32), fg=(255, 255, 255)):
    h, w = img.shape[:2]
    bar = np.full((bar_h, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, str(title), (10, int(bar_h*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2, cv2.LINE_AA)
    return np.vstack([bar, img])

def _panel(img, title, tile_size=(480, 360)):
    return _add_title_bar(_fit_letterbox(img, tile_size), title)

# =============================== MAIN LOOP =====================================
for image in images:
    print(image)
    img = cv2.imread(os.path.join(IMG_DIR, image))
    if img is None:
        print("No se pudo leer:", image)
        continue

    height, width = img.shape[:2]

    # Copias para dibujar
    img_mtcnn = img.copy()
    img_dlib  = img.copy()
    img_dnn   = img.copy()
    img_haar  = img.copy()

    # -------------------- # MTCNN --------------------
    t_pre0 = time.perf_counter()
    im_m, scale_m = preprocess(img.copy(), 'mtcnn')
    t_pre = (time.perf_counter() - t_pre0) * 1000

    t_det0 = time.perf_counter()
    if USE_SLICING['mtcnn']:
        faces1_boxes, faces1_scores = slice_and_detect(
            im_m, detect_mtcnn, tile=SLICE_CFG['tile'], overlap=SLICE_CFG['overlap'],
            mtcnn=detector1, conf_th=CONF['mtcnn']
        )
    else:
        faces1_boxes, faces1_scores = detect_mtcnn(im_m, detector1, conf_th=CONF['mtcnn'])
    t_det = (time.perf_counter() - t_det0) * 1000

    t_post0 = time.perf_counter()
    if USE_NMS['mtcnn']:
        faces1_boxes, faces1_scores = nms(faces1_boxes, faces1_scores, NMS_IOU['mtcnn'])
    faces1_boxes = descale_boxes(faces1_boxes, scale_m)
    faces1_boxes = clamp_boxes(faces1_boxes, width, height)
    t_post = (time.perf_counter() - t_post0) * 1000

    # -------------------- # DLIB ---------------------
    t_pre0 = time.perf_counter()
    im_d, scale_d = preprocess(img.copy(), 'dlib')
    t_pre_d = (time.perf_counter() - t_pre0) * 1000

    t_det0 = time.perf_counter()
    if USE_SLICING['dlib']:
        faces2_boxes, faces2_scores = slice_and_detect(
            im_d, detect_dlib_hog, tile=SLICE_CFG['tile'], overlap=SLICE_CFG['overlap'],
            hog=detector2, **HOG_CFG
        )
    else:
        faces2_boxes, faces2_scores = detect_dlib_hog(im_d, detector2, **HOG_CFG)
    t_det_d = (time.perf_counter() - t_det0) * 1000

    t_post0 = time.perf_counter()
    if USE_NMS['dlib']:
        faces2_boxes, faces2_scores = nms(faces2_boxes, faces2_scores, NMS_IOU['dlib'])
    faces2_boxes = descale_boxes(faces2_boxes, scale_d)
    faces2_boxes = clamp_boxes(faces2_boxes, width, height)
    t_post_d = (time.perf_counter() - t_post0) * 1000

    # -------------------- # DNN (OpenCV SSD) --------
    t_pre0 = time.perf_counter()
    im_n, scale_n = preprocess(img.copy(), 'dnn')
    t_pre_n = (time.perf_counter() - t_pre0) * 1000

    t_det0 = time.perf_counter()
    if USE_SLICING['dnn']:
        faces3_boxes, faces3_scores = slice_and_detect(
            im_n, detect_dnn, tile=SLICE_CFG['tile'], overlap=SLICE_CFG['overlap'],
            net=net, conf_th=CONF['dnn']
        )
    else:
        faces3_boxes, faces3_scores = detect_dnn(im_n, net, conf_th=CONF['dnn'])
    t_det_n = (time.perf_counter() - t_det0) * 1000

    t_post0 = time.perf_counter()
    if USE_NMS['dnn']:
        faces3_boxes, faces3_scores = nms(faces3_boxes, faces3_scores, NMS_IOU['dnn'])
    faces3_boxes = descale_boxes(faces3_boxes, scale_n)
    faces3_boxes = clamp_boxes(faces3_boxes, width, height)
    t_post_n = (time.perf_counter() - t_post0) * 1000

    # -------------------- # HAAR ---------------------
    t_pre0 = time.perf_counter()
    im_h, scale_h = preprocess(img.copy(), 'haar')
    t_pre_h = (time.perf_counter() - t_pre0) * 1000

    t_det0 = time.perf_counter()
    if USE_SLICING['haar']:
        faces4_boxes, faces4_scores = slice_and_detect(
            im_h, detect_haar, tile=SLICE_HAAR['tile'], overlap=SLICE_HAAR['overlap'],
            cascade=classifier2, **HAAR_CFG
        )
    else:
        faces4_boxes, faces4_scores = detect_haar(im_h, classifier2, **HAAR_CFG)
    t_det_h = (time.perf_counter() - t_det0) * 1000

    t_post0 = time.perf_counter()
    if USE_NMS['haar']:
        faces4_boxes, faces4_scores = nms(faces4_boxes, faces4_scores, NMS_IOU['haar'])
    faces4_boxes = descale_boxes(faces4_boxes, scale_h)
    faces4_boxes = clamp_boxes(faces4_boxes, width, height)
    t_post_h = (time.perf_counter() - t_post0) * 1000

    # =================== DIBUJO (cajas) ===================
    for (x1, y1, x2, y2) in faces1_boxes:
        cv2.rectangle(img_mtcnn, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for (x1, y1, x2, y2) in faces2_boxes:
        cv2.rectangle(img_dlib,  (x1, y1), (x2, y2), (0, 0, 255), 2)
    for (x1, y1, x2, y2) in faces3_boxes:
        cv2.rectangle(img_dnn,   (x1, y1), (x2, y2), (0, 0, 255), 2)
    for (x1, y1, x2, y2) in faces4_boxes:
        cv2.rectangle(img_haar,  (x1, y1), (x2, y2), (0, 0, 255), 2)

    # =================== LOGS ===================
    if LOG_TIMES:
        tot = t_pre + t_det + t_post
        logs_per_detector['mtcnn']['pre_ms'] += t_pre
        logs_per_detector['mtcnn']['det_ms'] += t_det
        logs_per_detector['mtcnn']['post_ms'] += t_post
        logs_per_detector['mtcnn']['total_ms'] += tot
        logs_per_detector['mtcnn']['count'] += 1

        tot = t_pre_d + t_det_d + t_post_d
        logs_per_detector['dlib']['pre_ms'] += t_pre_d
        logs_per_detector['dlib']['det_ms'] += t_det_d
        logs_per_detector['dlib']['post_ms'] += t_post_d
        logs_per_detector['dlib']['total_ms'] += tot
        logs_per_detector['dlib']['count'] += 1

        tot = t_pre_n + t_det_n + t_post_n
        logs_per_detector['dnn']['pre_ms'] += t_pre_n
        logs_per_detector['dnn']['det_ms'] += t_det_n
        logs_per_detector['dnn']['post_ms'] += t_post_n
        logs_per_detector['dnn']['total_ms'] += tot
        logs_per_detector['dnn']['count'] += 1

        tot = t_pre_h + t_det_h + t_post_h
        logs_per_detector['haar']['pre_ms'] += t_pre_h
        logs_per_detector['haar']['det_ms'] += t_det_h
        logs_per_detector['haar']['post_ms'] += t_post_h
        logs_per_detector['haar']['total_ms'] += tot
        logs_per_detector['haar']['count'] += 1

    # =================== VISUALIZACIÓN / MONTAJE 2x2 ===================
    if SHOW:
        p1 = _panel(img_mtcnn, "MTCNN")
        p2 = _panel(img_dlib,  "DLIB HOG")
        p3 = _panel(img_dnn,   "DNN (SSD-Caffe)")
        p4 = _panel(img_haar,  "HAAR")

        row1 = cv2.hconcat([p1, p2])
        row2 = cv2.hconcat([p3, p4])
        montage = cv2.vconcat([row1, row2])

        cv2.imshow("Comparativa de Detectores (2x2)", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # (Opcional) guardado del montaje
    if SAVE:
        cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(image)[0]}_mtcnn.jpg"), img_mtcnn)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(image)[0]}_dlib.jpg"),  img_dlib)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(image)[0]}_dnn.jpg"),   img_dnn)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(image)[0]}_haar.jpg"),  img_haar)
        # cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(image)[0]}_montage.jpg"), montage)

# =========================== RESUMEN DE LOGS ==============================
if LOG_TIMES:
    print("\n=== RESUMEN DE TIEMPOS (promedios por imagen) ===")
    for det, d in logs_per_detector.items():
        c = d['count']
        if c == 0:
            continue
        print(f"- {det.upper():5s} | pre: {d['pre_ms']/c:.1f} ms  det: {d['det_ms']/c:.1f} ms  "
              f"post: {d['post_ms']/c:.1f} ms  total: {d['total_ms']/c:.1f} ms  (n={c})")

    if SAVE_LOGS_CSV:
        try:
            import csv
            with open(LOGS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['detector', 'pre_ms_avg', 'det_ms_avg', 'post_ms_avg', 'total_ms_avg', 'n'])
                for det, d in logs_per_detector.items():
                    c = d['count']
                    if c == 0:
                        continue
                    w.writerow([det, d['pre_ms']/c, d['det_ms']/c, d['post_ms']/c, d['total_ms']/c, c])
            print(f"→ CSV con promedios guardado en: {LOGS_CSV_PATH}")
        except Exception as e:
            print("No se pudo guardar CSV:", e)
