import os
import re
import EncoderApi as face_recognition
import PIL.Image
import numpy as np
import math

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class DataSet:

    def __init__(self,data_folder_path):
        """

        :param data_folder_path:
        """
        self.num_jitters = 1
        self.face_encodings_model = "large"
        self.encodingModel="hog"  # hog or cnn
        self.data_folder_path = data_folder_path
        self.faces = self.__init_training_data() if data_folder_path is not None else None

    def __image_files_in_folder__(self,folder):
        """

        :param folder:
        :return:
        """
        return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

    def __load_image_file__(self,file, mode='RGB'):
        """

        :param file:
        :param mode:
        :return:
        """
        im = PIL.Image.open(file)
        if mode:
            im = im.convert(mode)
        return np.array(im)


    def __init_training_data(self,n_neighbors=None):
        """
            :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
            :param model: face_recognition defualt is hog, which is faster but lesser accurate . other values is cnn.
                          cnn is more accurate but the slower than hog
            :return: labels and identified faces
        """
        print ("Started image encoding ...")
        pathjoin = os.path.join
        faces = []
        labels = []  # list to hold all subject faces

        # let's go through each directory and read images within it
        for dir_name in os.listdir(self.data_folder_path):

            if not os.path.isdir(pathjoin(self.data_folder_path, dir_name)):
                continue

            for img_path in self.__image_files_in_folder__(pathjoin(self.data_folder_path, dir_name)):
                image = self.__load_image_file__(img_path)
                face_bounding_boxes = face_recognition.face_locations(image,model= self.encodingModel)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    faceEncoding = face_recognition.face_encodings(image, num_jitters =self.num_jitters , model=self.face_encodings_model)
                    if faceEncoding is not None and len(faceEncoding) > 0 :
                        faces.append(faceEncoding[0])
                        labels.append(dir_name)
                    else :
                        print ("unbale to encode the image ",img_path)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(faces))))
            print("Chose n_neighbors automatically:", n_neighbors)
        return faces,labels

    ''' 
    Return the encoded faces and its labels 
    '''
    def getdata(self):
        """

        :return:
        """
        return self.faces[0], self.faces[1]

    def get_faces_encodings(self,rgbframe,number_of_times_to_upsample=1):
        """

        :param img_path:
        :param number_of_times_to_upsample:
        :return:
        """
        # Load image file and find face locations
        #X_img = self.__load_image_file__(img_path)
        X_face_locations = face_recognition.face_locations(rgbframe,number_of_times_to_upsample = 1,model= self.encodingModel)
        # Find encodings for faces in the image
        return face_recognition.face_encodings(rgbframe, known_face_locations=X_face_locations,num_jitters =1,model=self.face_encodings_model) , X_face_locations