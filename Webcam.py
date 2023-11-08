import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
class GestureRecognition:
    def main(self):
        #GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = 'C:\\Users\\sethl\\OneDrive\\Documents\\Senior Design Proj\\gesture_recognizer.task'

        self.lock = threading.Lock()
        self.currentGesture = []
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        video = cv2.VideoCapture(0)

        while (True):
            frameGrab, frame = video.read()
            if not frameGrab:
                break
            flipped_frame = cv2.flip(frame, 1)
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped_frame)
            recognizer.recognize_async(mp_image, int(timestamp))
            self.frame_gesture(frame)
            cv2.imshow("Gesture Interpreter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()

    def frame_gesture(self, frame):
        self.lock.acquire()
        gestures = self.currentGesture
        self.lock.release()
        for name in gestures:
            cv2.putText(frame, name, (225, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            cv2.putText(frame, name, (225, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def __result_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        self.lock.acquire()
        self.currentGesture = []
        for hand in result.handedness:
            print("Hand:")
            handName = hand[0].category_name
            print(handName)
        for hand_gesture in result.gestures:
            print("Gestures:")
            gestureName = hand_gesture[0].category_name
            gesturePercent = str(format(f"{hand_gesture[0].score:.2%}"))
            print(gestureName)
            self.currentGesture.append(gestureName + " " + gesturePercent)
        self.lock.release()
        # print('gesture recognition result: {}'.format(result))

if __name__ == "__main__":
    rec = GestureRecognition()
    rec.main()
