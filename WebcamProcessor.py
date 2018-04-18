import cv2
from threading import Thread


class Processor(Thread):

    def __init__(self,name,cam_url,model):
        Thread.__init__(self)
        self.threadName = name
        self.cam_url = cam_url
        self.model = model

    def run(self):
        self.process()

    def process(self):
        video_capture = cv2.VideoCapture(self.cam_url)
        process_this_frame = True
        predictions = []
        while True and video_capture.isOpened():

            # Grab a single frame of video
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                predictions = self.model.predictFromImageFrame(rgb_frame)

            process_this_frame = not process_this_frame
            self.drowText(frame, predictions)

            # Display the resulting image
            cv2.imshow(self.threadName, frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def drowText(self,frame,predictions):
        for name, (top, right, bottom, left) in predictions:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
