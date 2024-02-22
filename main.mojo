from python import Python



def main():
    Python.add_to_path(".")
    let cv2 = Python.import_module('cv2')
    let yolo = Python.import_module('ultralytics')
    let sv = Python.import_module('supervision')
    let rf = Python.import_module('roboflow')
    let inference = Python.import_module('inference')

    let stream_url = "http://192.168.68.107:8080/video"
    # let model = yolo.YOLO("best.pt")
    let cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Cannot open stream")
        return

    let rf_p = rf.Roboflow("sNdwlomV5jCTZh3j53zm")
    let project = rf_p.workspace("qwerty-v5soj").project("svalik")
    let model = inference.get_roboflow_model("svalik/1")


    while cap.isOpened():
        var qwer = cap.read()
        var success = qwer[0]
        var frame = qwer[1]
        let filename = 'qwerty.png'
        

        if not success:
            print("Can't receive frame. Exiting ...")
        else:
            
            cv2.imwrite(filename, frame)
            print(filename)
        cap.release()

        var image = cv2.imread(filename)

        if success:
            try:
                var results = model.infer(frame)
                let detections = sv.Detections.from_inference(results[0])
                let bounding_box_annotator = sv.BoundingBoxAnnotator()
                let label_annotator = sv.LabelAnnotator()
                var annotated_image = bounding_box_annotator.annotate(image, detections)
                annotated_image = label_annotator.annotate(annotated_image, detections)
                # var annotated_frame = results[0].plot()
                # cv2.imshow("YOLOv8 Inference", annotated_frame)
                sv.plot_image(annotated_image)
            except Exeption:
                print(Exeption)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    # cap.release()
    # cv2.destroyAllWindows()