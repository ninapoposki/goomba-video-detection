import cv2
import numpy as np
import os
import sys

def load_video_counts(csv_path):
    actual_counts = {} 
    with open(csv_path, 'r') as file: 
        lines = file.readlines()  
        for line in lines[1:]: 
            video_name, count = line.strip().split(',')  
            actual_counts[video_name] = int(count) 
    return actual_counts

LOWER_BROWN1 = np.array([10, 100, 50])
UPPER_BROWN1 = np.array([18, 255, 200])

LOWER_BROWN2 = np.array([0, 50, 50])  
UPPER_BROWN2 = np.array([10, 200, 200])

kernel = np.ones((8, 8), np.uint8) 

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    total_detected_objects = 0 
    frame_count = 0
    line_x_left = None 
    line_x_right = None  
    paused = False
    
    skip_frames =1
    while True: 
        
        if not paused:
            
            for _ in range(skip_frames): 
                cap.grab() 
            skip_frames =0
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count == 1:
                width = frame.shape[1] 
                line_x_left = int(width * 0.463)  
                line_x_right = int(width * 0.478)  

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, LOWER_BROWN1, UPPER_BROWN1)
            mask2 = cv2.inRange(hsv, LOWER_BROWN2, UPPER_BROWN2)

            combined_mask = cv2.bitwise_or(mask1, mask2)

            combined_mask = cv2.erode(combined_mask, kernel, iterations=2)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            stacked = 0 
            for cnt in contours:
                area = cv2.contourArea(cnt) 
                if area > 350: 
                    x, y, w, h = cv2.boundingRect(cnt) 
                    #x i y koordinate gornjeg levog ugla 
                    #w i h sirina i visina pravougaonika
                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

                    if (line_x_left < x < line_x_right) :
                        total_detected_objects += 1
                        #print(f"prosao:{total_detected_objects} ")
                        stacked+=1
                        skip_frames = 2 
            
            cv2.line(frame, (line_x_left, 0), (line_x_left, frame.shape[0]), (255, 0, 0), 2)  
            cv2.line(frame, (line_x_right, 0), (line_x_right, frame.shape[0]), (0, 0, 255), 2)  

        resized_frame = cv2.resize(frame, (640, 360)) 
        resized_mask = cv2.resize(combined_mask, (640, 360))  

        cv2.imshow("Processed Frame",frame)
        # # cv2.imshow("Mask", combined_mask)
         
        key = cv2.waitKey(10)

        if key == ord('q'):  
            break
        elif key == ord('p'):  
            paused = not paused 

    cap.release()
    cv2.destroyAllWindows()
    return total_detected_objects
  

def evaluate_results(video_folder, csv_path):
    true_counts = load_video_counts(csv_path)
    results = []

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    # sorted_files = []
    # for f in video_files:
    #     underscore_index = f.index('_') + 1
    #     dot_index = f.index('.')
    #     number = int(f[underscore_index:dot_index])
    #     sorted_files.append((number, f))
    # sorted_files.sort()
    # video_files = [f[1] for f in sorted_files]  

    for video_file in video_files:
        video_name = video_file
        # print(f"Na redu {video_name}")

        predicted_count = process_video(os.path.join(video_folder, video_file))
        actual_count = true_counts.get(video_name, np.nan)
        results.append({'video': video_name, 'predicted': predicted_count, 'actual': actual_count})
        # print(f"Video: {video_name}, Predviđeno: {predicted_count}, Tačno: {actual_count}")

    total_error = sum(abs(r['predicted'] - r['actual']) for r in results)
    mae = total_error / len(results)    
    # print(mae)
    return mae

def main(folder_path):
    video_folder = folder_path
    csv_path = os.path.join(folder_path, 'goomba_count.csv')
    mae = evaluate_results(video_folder, csv_path)
    print(mae)

folder_path = sys.argv[1]
main(folder_path)
