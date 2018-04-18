from sklearn import neighbors
import pickle
import os
from DataSet import DataSet


class Model :

    def __init__(self,dataset, n_neighbors=1,knn_algo='ball_tree',trained =False):
        self.__model_save_path= "dlibmodels/knnModel.clf"
        self.model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        self.__load_Model__() if trained else self.train(dataset)

    def __load_Model__(self):
        if self.__model_save_path is not None:
            with open(self.__model_save_path, 'rb') as f:
                self.model = pickle.load(f)

    def incrementalTrain(self,dataset):
        x, y = dataset.getdata()
        self.model.fit(x, y)
        # Save the trained KNN classifier
        if self.__model_save_path is not None:
            with open(self.__model_save_path, 'wb') as f:
                pickle.dump(self.model, f)

    def train(self,dataset):
        print ("start training classifier")
        x,y = dataset.getdata()
        self.model.fit(x, y)
        # Save the trained KNN classifier
        if self.__model_save_path is not None:
            with open(self.__model_save_path, 'wb') as f:
                pickle.dump(self.model, f)
        print ("Training completed")

    def predict(self,img_path, distance_threshold=0.7):
        """
            Recognizes faces in given image using a trained KNN classifier
            :param X_img_path: path to image to be recognized
            :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
                   of mis-classifying an unknown person as a known one.
            :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
                For faces of unrecognized persons, the name 'unknown' will be returned.
            """
        faces_encodings, X_face_locations = DataSet(None).get_faces_encodings(img_path)
        # Use the KNN model to find the best matches for the test face
        closest_distances = self.model.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(self.model.predict(faces_encodings), X_face_locations, are_matches)]

    def predictFromImageFrame(self, imageFrame, distance_threshold=0.5):
        """
            Recognizes faces in given image using a trained KNN classifier
            :param X_img_path: path to image to be recognized
            :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
                   of mis-classifying an unknown person as a known one.
            :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
                For faces of unrecognized persons, the name 'unknown' will be returned.
            """
        faces_encodings, X_face_locations = DataSet(None).get_faces_encodings(imageFrame)
        # Use the KNN model to find the best matches for the test face
        if X_face_locations is not None and len(X_face_locations) > 0 :
            closest_distances = self.model.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                    zip(self.model.predict(faces_encodings), X_face_locations, are_matches)]
        return []


