# main.py
# RDK X5 Garbage Detection System


from camera import Camera
from detect import GarbageDetector
from utils import draw_result, FPSCounter

import cv2



def main():


    print("==============================")
    print(" RDK X5 Garbage AI System ")
    print("==============================")


    camera = Camera(
        camera_id=0
    )


    detector = GarbageDetector(

        model_path="../model/garbage_yolov8.bin"

    )


    fps = FPSCounter()



    while True:


        frame = camera.read()


        if frame is None:

            continue



        result = detector.detect(
            frame
        )


        frame = draw_result(

            frame,

            result

        )


        fps.update()



        cv2.putText(

            frame,

            fps.get(),

            (20,40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0,255,0),

            2

        )



        cv2.imshow(

            "RDK X5 Garbage Detection",

            frame

        )



        key=cv2.waitKey(1)



        if key==ord("q"):

            break



    camera.release()



if __name__=="__main__":

    main()