import json
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
    '2019-05-7',
    iam_apikey='L9k7HYkIssnQ-rrT7_oEWQUJAbyyzKSPv0n_VKfKA-M8')

test_url = 'http://192.168.1.8:8080/shot.jpg' #Contains the link to the IPWebcam stream
faces = visual_recognition.detect_faces(parameters=json.dumps({'url': test_url}))
#faces = visual_recognition.detect_faces(url).get_result()
print(json.dumps(faces, indent=2))