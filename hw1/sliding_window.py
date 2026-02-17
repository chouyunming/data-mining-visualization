import cv2
import numpy as np

# 定義路徑
target = 'space2'
template_path = f'C:/Users/ander/Downloads/113-1_DMVL_Project1/minions_original.png'
target_path = f'C:/Users/ander/Downloads/113-1_DMVL_Project1/detect_{target}.png'

# 讀取圖片
template_img = cv2.imread(template_path)
target_img = cv2.imread(target_path)

# 將範例圖片和目標圖片轉換為灰階
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# 設置匹配方法和相似度閾值
method = cv2.TM_CCOEFF_NORMED
similarity_threshold = 0.6

# 假設小小兵的基礎大小約為 50x60，針對不同大小設定 bounding box
base_width, base_height = 50, 60
bounding_boxes = [
    (int(base_width * 0.2), int(base_height * 0.2)),
    (int(base_width * 0.4), int(base_height * 0.4)),
    (int(base_width * 0.6), int(base_height * 0.6)),
    (int(base_width * 0.8), int(base_height * 0.8)),
    (int(base_width), int(base_height)),
    (int(base_width * 1.2), int(base_height * 1.2)),
    (int(base_width * 1.5), int(base_height * 1.5)),
    (int(base_width * 1.8), int(base_height * 1.8)),
    (int(base_width * 2.0), int(base_height * 2.0)),
    (int(base_width * 3.0), int(base_height * 3.0))
]

# 儲存所有偵測到的框
detected_objects = []

# 對於每個 bounding box 尺寸，進行模板匹配
for (box_width, box_height) in bounding_boxes:
    # 縮放模板到指定大小
    resized_template = cv2.resize(template_gray, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
    
    # 進行模板匹配
    result = cv2.matchTemplate(target_gray, resized_template, method)
    
    # 找出高於閾值的位置
    locations = np.where(result >= similarity_threshold)
    
    # 儲存每個符合條件的框
    for pt in zip(*locations[::-1]):
        detected_objects.append({
            "top_left": (pt[0], pt[1]),
            "width": box_width,
            "height": box_height,
            "score": result[pt[1], pt[0]]
        })

# 非極大值抑制（NMS）來移除重疊的框
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    # 初始化框的座標
    boxes = np.array([[box['top_left'][0], box['top_left'][1], box['top_left'][0] + box['width'], box['top_left'][1] + box['height'], box['score']] for box in boxes])
    
    # 建立框的左上和右下座標
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # 計算每個框的面積並排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 根據分數降序排序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 計算重疊區域的範圍
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 計算重疊面積
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]

        # 保留重疊比例小於閾值的框
        order = order[np.where(overlap <= overlap_thresh)[0] + 1]

    # 依據篩選結果返回過濾後的框
    return boxes[keep].astype(int)

# 使用 NMS 過濾重疊的框
nms_boxes = non_max_suppression(detected_objects)

# 在目標圖片上繪製 NMS 過後的方框
for (x1, y1, x2, y2, _) in nms_boxes:
    cv2.rectangle(target_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 儲存結果圖片
output_path = f'C:/Users/ander/Downloads/113-1_DMVL_Project1/detection_output/{target}.png'
cv2.imwrite(output_path, target_img)

# 顯示檢測到的結果圖片路徑
print(f"Detection result saved at: {output_path}")

# 顯示結果
cv2.imshow('Detection Result', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()