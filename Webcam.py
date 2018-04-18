import EncoderApi as  face_recognition
import cv2
from DataSet import DataSet


dataset = DataSet("../data/training-data/incremental")
known_face_encodings, known_face_names = dataset.getdata()



#video_capture = cv2.VideoCapture("rtsp://admin:123456@192.168.43.180:554")
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
'''obama_image = face_recognition.load_image_file("../data/training-data/incremental/shijujohn.jpeg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("../data/training-data/incremental/akt.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
athul_image = face_recognition.load_image_file("../data/training-data/incremental/AthulRaj.jpg")
athul_face_encoding = face_recognition.face_encodings(athul_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    athul_face_encoding
]

known_face_names = [
    "Shiju John",
    "Arun Kumar T",
    "Athul Raj"
]'''




#face_recognition.face_encodings(biden_image)[0]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        dataset = DataSet(None,)
        face_names = []
        face_encodings, face_locations  = dataset.get_faces_encodings(rgb_frame)
        #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)


        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.45)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names [first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
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

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()