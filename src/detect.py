# detect.py
# YOLOv8 + RDK X5 .bin garbage detection


import cv2
import numpy as np



class GarbageDetector:


    def __init__(
            self,
            model_path,
            conf_threshold=0.5,
            nms_threshold=0.45
    ):


        self.model_path = model_path

        self.conf_threshold = conf_threshold

        self.nms_threshold = nms_threshold


        self.classes = [

            "可回收垃圾",

            "厨余垃圾",

            "有害垃圾",

            "其他垃圾"

        ]


        self.model = None


        self.input_size = 640


        self.load_model()



    # ==========================
    # 加载RDK X5模型
    # ==========================

    def load_model(self):

        try:

            from hbm_runtime import HbmRuntime


            self.model = HbmRuntime(
                self.model_path
            )


            print(
                "RDK X5 bin model loaded!"
            )


        except Exception as e:


            print(
                "Model load failed:",
                e
            )



    # ==========================
    # YOLOv8预处理
    # ==========================

    def preprocess(
            self,
            image
    ):


        self.orig_h, self.orig_w = image.shape[:2]


        img = cv2.resize(

            image,

            (
                self.input_size,
                self.input_size
            )

        )


        img = cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )


        img = np.expand_dims(

            img,

            axis=0

        )


        img = img.astype(
            np.uint8
        )


        return img



    # ==========================
    # RDK X5 推理
    # ==========================

    def inference(
            self,
            input_data
    ):


        if self.model is None:

            return None



        try:


            output = self.model.run(

                input_data

            )


            return output



        except Exception as e:


            print(
                "Inference error:",
                e
            )


            return None




    # ==========================
    # NMS
    # ==========================

    def nms(
            self,
            boxes,
            scores
    ):


        index = cv2.dnn.NMSBoxes(

            boxes,

            scores,

            self.conf_threshold,

            self.nms_threshold

        )


        if len(index)==0:

            return []


        return index.flatten()



    # ==========================
    # YOLOv8后处理
    # ==========================

    def postprocess(
            self,
            output
    ):


        if output is None:

            return None



        # hbm_runtime可能返回list

        if isinstance(output,list):

            pred = output[0]


        else:

            pred = output



        pred=np.array(pred)



        # 去除batch维

        if pred.ndim==3:

            pred=pred[0]



        print(
            "YOLO output shape:",
            pred.shape
        )



        # ======================
        # 自动判断YOLO格式
        # ======================


        # 情况1:
        # (8,8400)

        if pred.shape[0] < pred.shape[1]:

            pred = pred.T



        # 当前:
        # (8400,8)


        boxes=[]

        scores=[]

        class_ids=[]



        for det in pred:



            # x,y,w,h

            x,y,w,h = det[:4]



            cls_scores = det[4:]



            class_id = np.argmax(

                cls_scores

            )


            confidence = cls_scores[class_id]



            if confidence < self.conf_threshold:

                continue



            # 坐标恢复


            x1 = int(

                (x-w/2)

                *

                self.orig_w

                /

                self.input_size

            )


            y1 = int(

                (y-h/2)

                *

                self.orig_h

                /

                self.input_size

            )


            x2 = int(

                (x+w/2)

                *

                self.orig_w

                /

                self.input_size

            )


            y2 = int(

                (y+h/2)

                *

                self.orig_h

                /

                self.input_size

            )



            boxes.append(

                [

                    x1,

                    y1,

                    x2-x1,

                    y2-y1

                ]

            )


            scores.append(

                float(confidence)

            )


            class_ids.append(

                int(class_id)

            )



        if len(boxes)==0:

            return None



        keep=self.nms(

            boxes,

            scores

        )


        if len(keep)==0:

            return None



        # 取最高置信度目标

        best=keep[0]



        return {


            "class":

            self.classes[
                class_ids[best]
            ],



            "confidence":

            scores[best],



            "box":

            [

                boxes[best][0],

                boxes[best][1],

                boxes[best][0]+boxes[best][2],

                boxes[best][1]+boxes[best][3]

            ]

        }




    # ==========================
    # 外部调用接口
    # ==========================

    def detect(
            self,
            image
    ):


        input_data=self.preprocess(

            image

        )


        output=self.inference(

            input_data

        )


        result=self.postprocess(

            output

        )


        return result