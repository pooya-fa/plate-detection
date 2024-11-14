
import torch
import cv2
import numpy as np
import time


#function to run detection
def detectx(frame, model):
    frame = [frame]
    print(" Detecting. . . ")
    results = model(frame)
    results.show()
    print(results)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


###Main function

def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    # loading the custom trained model

    model = torch.hub.load('E:\py\BloodCels\yolov5', 'custom',
                           source='local', path='best.pt', force_reload=True)

    classes = model.names  ### class names in string format

    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)  # DETECTION HAPPENING HERE

        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  # creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                print(f"[INFO] Exiting. . . ")
                # cv2.imwrite("final_output.jpg", frame)  ## if you want to save he output result.

                break

    elif vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")

        # reading the video
        cap = cv2.VideoCapture(vid_path)

        if vid_out:  # creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')  #(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                frame_no += 1

        print(f"[INFO] Clening up. . . ")
        # releaseing the writer
        out.release()

        # closing all windows
        cv2.destroyAllWindows()


#calling the main function


# main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") # for custom video
# main(vid_path=0,vid_out="webcam_facemask_result.mp4") # for webcam

main(img_path="E:\py\plate_detection\h2.jpg") # for image

