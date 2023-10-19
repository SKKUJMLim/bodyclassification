import cv2
import os
import json
from collections import OrderedDict

FRAME_PATH = 'D:/data/Resize_frames/'
JSON_SAVE_PATH = 'D:/data/Openpose_json/'
IMAGE_SAVE_PATH = 'D:/data/Openpose_images/' 

BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                  [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_mpi = "models\\pose\\mpi\\pose_deploy_linevec.prototxt"
protoFile_mpi_faster = "models\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
protoFile_coco = "models\\pose\\coco\\pose_deploy_linevec.prototxt"
protoFile_body_25 = "models\\pose\\body_25\\pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_mpi = "models\\pose\\mpi\\pose_iter_160000.caffemodel"
weightsFile_coco = "models\\pose\\coco\\pose_iter_440000.caffemodel"
weightsFile_body_25 = "models\\pose\\body_25\\pose_iter_584000.caffemodel"

def output_keypoints(frame, frame_path, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x= round(x,3)
        y = (frame_height * point[1]) / out_height
        y= round(y,3)
        prob= round(prob, 6)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append(x)
            points.append(y)
            points.append(prob)
            #print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(x)
            points.append(y)
            points.append(prob)
            #print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    
    # cv2.imshow("Output_Keypoints", frame)
    
    if not os.path.exists(IMAGE_SAVE_PATH):
            os.makedirs(IMAGE_SAVE_PATH)
    
    
    save_path = IMAGE_SAVE_PATH + os.path.splitext(frame_path)[0]+".jpg"
    cv2.imwrite(save_path, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
          
    return points
    
def output_keypoints_with_lines(frame, POSE_PAIRS):
    print()
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    cv2.imshow("output_keypoints_with_lines", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":

    for frame_path in os.listdir(FRAME_PATH):
    
        print("Current : " + FRAME_PATH +  frame_path)
            
        img = cv2.imread(FRAME_PATH + frame_path)
        frame_body_25 = img.copy()

        # BODY_25 Model
        points = output_keypoints(frame=frame_body_25, frame_path=frame_path, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                     threshold=0.2, model_name="BODY_25", BODY_PARTS=BODY_PARTS_BODY_25)

        file_Data = OrderedDict()
        file_Data["version"] = 1.3
        
        # Background는 제거
        file_Data["people"] = [{'person_id' : [-1], 'pose_keypoints_2d': points[:-3], 'face_keypoints_2d':[],'hand_left_keypoints_2d':[],'pose_keypoints_3d':[],'face_keypoints_3d':[],'hand_left_keypoints_3d':[],'hand_right_keypoints_3d':[]}]
        
        # JSON 파일 생성
        
        if not os.path.exists(JSON_SAVE_PATH):
            os.makedirs(JSON_SAVE_PATH)
        
        save_path = JSON_SAVE_PATH + os.path.splitext(frame_path)[0]+".json"
        
        with open(save_path, 'w', encoding='utf-8') as make_file:
            json.dump(file_Data, make_file, ensure_ascii=False, indent='\t')