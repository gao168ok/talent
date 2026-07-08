# camera.py

import cv2



class Camera:



    def __init__(
            self,
            camera_id=0
    ):


        self.cap=cv2.VideoCapture(
            camera_id
        )


        if not self.cap.isOpened():

            raise Exception(
                "Camera open failed"
            )



    def read(self):


        ret,frame=self.cap.read()



        if ret:

            return frame


        return None




    def release(self):


        self.cap.release()

        cv2.destroyAllWindows()