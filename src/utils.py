# utils.py


import cv2
import time




def draw_result(

        frame,

        result

):



    if result is None:

        return frame



    text=(

        result["class"]

        +

        " "

        +

        str(

            round(

                result["confidence"],

                2

            )

        )

    )



    box=result["box"]



    cv2.rectangle(

        frame,

        (

            box[0],

            box[1]

        ),

        (

            box[2],

            box[3]

        ),

        (0,255,0),

        2

    )



    cv2.putText(

        frame,

        text,

        (

            box[0],

            box[1]-10

        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0,255,0),

        2

    )



    return frame





class FPSCounter:



    def __init__(self):

        self.last=time.time()

        self.fps=0




    def update(self):


        now=time.time()


        self.fps=1/(now-self.last)


        self.last=now




    def get(self):


        return "FPS:"+str(

            int(self.fps)

        )