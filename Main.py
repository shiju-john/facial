
from DataSet import DataSet
from FaceClassificationModel import  Model
from WebcamProcessor import  Processor
import sys

training_dataSet_url = '../data/training-data/incremental'
camera_urls = {
    #"camera_1":0,
    "camera_2":"rtsp://admin:123456@192.168.127.122:554"
}


threads =[]
if __name__ == "__main__":

    print ("*** If the nurel net is already trained -L or  -l to load from disk ***")
    trained = False
    for arg in sys.argv :
        if arg=='-l' or arg == '-L': trained = True
    dataset = DataSet(training_dataSet_url) if not trained else None
    model = Model(dataset, trained=trained)
    for key, value in camera_urls.viewitems() :
       thread = Processor(key,value,model)
       thread.start()
       threads.append(thread)



