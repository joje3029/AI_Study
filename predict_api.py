#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:58:02 2023 만든 날짜

@author: henry 만든 놈
"""


from flask import Flask, render_template, Response, request #플라스크 사용하기위해
import json #json 사용하기 위해
import argparse #명령행 인자를 다루기 위해 argparse 모듈 가져옴
import os #시스템 관련 모듈
import sys #파일 관련 모듈
from pathlib import Path #경로를 다루기 위해 path 모듈

from ultralytics import YOLO # 울트라틱스에서 YOLo 
from ultralytics.utils.checks import cv2, print_args #울트라틱스에서 제공하는 몇 가지 유틸리티 함수들을 가져옴
# from utils2.general import update_options #utils2 모듈에서 update_option 함수를 가져옴.
from utils.general import update_options

# Initialize paths : 경로 초기화
FILE = Path(__file__).resolve() #현재 파일의 절대 경로를 얻음
ROOT = FILE.parents[0] #FILE 의 상위 디렉토리를 찾아 Root 변수에 할당함.
if str(ROOT) not in sys.path: # 만약 root 경로가 시스템 경로에 없으면
    sys.path.append(str(ROOT)) #시스템 경로에 추가한다 루트경로를
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) #경로가 있으면 루트 경로를 현재 작업 디렉토리 기준으로 한 상대 경로로 변환함. 
# relpath() : 경로를 상대 경로로 변환하는 함수. 

# Initialize Flask API : 플라스크 초기화 -> Flask를 사용하여 API를 작성하고 구성하고 있음. 이 API는 YOLO 모델을 사용하여 객체 감지를 수행하고 결과를 반환함.
app = Flask(__name__) #flask 애플리케이션을 생성할때 현재 모듈의 이름을 이용해서 생성한 어플리케이션이 어디서 실행 되는지 위치를 안다.


def predict(opt): #예측하는 일을 하는 함수.
    """
    Perform object detection using the YOLO model and yield results.
    # YOLO 모델을 사용하여 객체 검출을 수행하고 결과를 생성함.
    
    Parameters: 
    #매개변수
    - opt (Namespace): A namespace object that contains all the options for YOLO object detection,
        including source, model path, confidence thresholds, etc.
    #YOLO 객체 검출에 필요한 모든 옵션을 담고 있는 네임스페이스 객체. 여기에는 소스, 모델 경로, 신뢰도 임계값 등이 포함됨.
        
    Yields:
    #생성물
    - JSON: If opt.save_txt is True, yields a JSON string containing the detection results.
    # JSON : opt.save_txt가 True인 경우 검출 결과를 담은 JSON 문자열을 생성함.
    - bytes: If opt.save_txt is False, yields JPEG-encoded image bytes with object detection results plotted.
    # bytes : opt.save_txt가 False인 경우 객체 검출 결과를 포함한 JPEG로 인코딩된 이미지 바이트를 생성함.
    """
    
    
    results = model(**vars(opt), stream=True)
    #**vars(opt) : 파이썬에서 사용되는 딕셔너리를 풀어서(언패킹하여) 함수의 키워드 인자로 전달하는 특별한 문법
    # vars() : __dict__ 속성을 반환함. __dict__는 객체의 속성들을 담고 있는 딕셔너리. 
    #얻은 딕셔너리를 함수 호출시 **를 사용하여 풀어서 전달하면, 딕셔너리의 키-값 쌍들이 함수의 키워드 인자로 전달.
    # ** :  파이썬에서 언패킹 연산자로 사용. 함수 호출이나 컨테이너(리스트, 튜플, 딕셔너리 등)를 확장할 때 사용됨.

    for result in results: # 언패킹한걸 for문으로 푸는 중
        if opt.save_txt: #만약 네임스페이스 객체(opt)에 txt가 저장되어있으면
            result_json = json.loads(result.tojson()) # result의 json을 json형식으로 로드해서 result_json에 담는다.
            yield json.dumps({'results': result_json}) # 키를 results로 해서 result_json에 담긴 변수를 json형태에 추가한다.
        else: # txt가 저장되어있지 않으면
            im0 = cv2.imencode('.jpg', result.plot())[1].tobytes() # result 객체의 plot() 메서드를 호출하여 객체 탐지 결과를 시각화한 이미지를 얻음. 위에서 얻은 바이트 배열을 가져와 tobytes() 메서드를 사용하여 바이트로 변환함.
            yield (b'--frame\r\n' # 각 프레임의 시작을 나타내는 boundary 문자열
                   b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n') #이미지 데이터의 타입을 지정하는 부분.
                # im0 : JPEG로 인코딩된 이미지의 바이트 데이터가 여기에 들어감.
                # b'\r\n' : 각 프레임의 끝을 나타내는 구분자.
                # => Flask 애플리케이션이 클라이언트에게 한 번에 하나의 이미지 프레임을 스트리밍하여 전송하는 일반적인 방식 중 하나.
                # 이를 통해 실시간으로 객체 감지 결과를 시각적으로 확인가능

@app.route('/') #app의 경로가 루트일때 
def index(): #인덱스함수 리턴 -> 
    """
    Video streaming home page.
    비디오 스트리밍 홈페이지
    """
    # render_template : Flask 애플리케이션에서 HTML템플릿을 렌더링 하는 데 사용되는 함수/
    return render_template('index.html') # 현재 디렉토리에서 templates 폴더내에 있는 index.html 파일을 찾아 렌더링 함.


@app.route('/predict', methods=['GET', 'POST']) #app의 경로가 predict이고 get/post 방법일 때
def video_feed(): #video_feed 함수
    if request.method == 'POST': #만약에 post 방식이면 
        uploaded_file = request.files.get('myfile') #request 에서 myfile이라는 이름의 파일을 꺼내서 uploaded_file 변수에 담는다. 
        # * myfile 은 클라이언트에서 요청 보낼때 key 이름이 myfile
        save_txt = request.form.get('save_txt', 'F')  # Default to 'F' if save_txt is not provided : 만약 save_txt가 제공되지 않으면 기본값 F
        # * 클라이언트에서 save_txt가 파라미터로 설정되어있음. 
        if uploaded_file: #만약에  uploaded_file이면
            source = Path(__file__).parent / raw_data / uploaded_file.filename # url 만드는 중 : file의 부모디렉토리 + raw_data+업로드할 파일.파일타입
            uploaded_file.save(source) #위의 url을 uploaded_file에 저장
            opt.source = source #opt의 source에 위에서 만든 경로 저장
        else: # uploaded_file이 아니면
            opt.source, _ = update_options(request) #opt의 source에 걍 기존의 request를 넣음
            
        opt.save_txt = True if save_txt == 'T' else False #파이썬 3항 연산자 사용 : save_txt가 t이면 true아니면 false가 opt.save_txt 에 할당
            
    elif request.method == 'GET': # 만약 get방식이다.
        opt.source, opt.save_txt = update_options(request) #그럼 굳이 위의 저런짓 안하고 요청온거 고대로 opt.source와 opt.save_txt에 맞춰서 때림

    # Response 객체를 통해 클라이언트에게 반환함.
    return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')
    # mimetype : 전송되는 데이터의 형식을 나타내는 것. 
    # mimetype='multipart/x-mixed-replace; boundary=frame' : 여러 부분으로 나뉜 데이터를 사용. 각 부분은 frame 이라는 경계로 구분됨. 이러한 형식은 이미지 스트리밍과 같이 여러 이미지를 한 번에 전송하는데 사용됨.


if __name__ == '__main__': # name이 main이면 
    # Input arguments # 밑의 arguments가 실행
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default=ROOT / 'yolov8s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf','--conf-thres', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show','--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt','--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels','--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det','--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true', help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int, help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list, help='filter results by class, i.e. classes=0, or classes=[0,2,3]') # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()
    # parse_known_args() : 명령해 인자를 파싱하는데 사용되는 argparse메서드 => 알려지지 않은 인자들을 무시하고 남겨둠.
    # argparse는 에러를 발생시키지만 parse_known_args()는 에러를 발생시키지 않고 남은 인자들을 무시함.


    # print used arguments : 사용하는 arguments 프린트
    print_args(vars(opt))

    # Get por to deploy : 
    port = opt.port # 포트번호를 가져와서 port 변수에 할당.
    delattr(opt, 'port') #delattr() : 객체의 속성을 삭제하는데 사용. delattr(object, name) : object = 속성을 삭제할 객체 / name = 삭제할 속성의 이름.
    
    # Create path for raw data : raw data경로 만들기
    raw_data = Path(opt.raw_data) #rqw_data 변수에 opt의 raw_data 경로를 담음.
    raw_data.mkdir(parents=True, exist_ok=True) # raw_data 폴더를 만드는데 필요하면 부모폴더도 만들고 이미 폴더가 존재하면 에러를 발생시키지 않도록 설정한것.
    delattr(opt, 'raw_data') #opt 객체에서 raw-data 속성을 제거하고 있음. 속성을 제거하면 해당 속성에 대한 정보가 삭제되므로 이후에는  opt.raw_data를 사용할 수 없게 됨.
    
    # Load model (Ensemble is not supported) : 아래에서 사용한 YOLO()는 단일 YOLO모델만을 초기화하고 로드하는 기능을 제공해서 주인장이 이렇게 적었을 확률이 높음.
    model = YOLO(str(opt.model)) #opt의 모델의 타입을 String으로 해서 YOLO모델을 초기화하는 함수.
    #YOLO() : Ultralytics 라이브러리에서 제공하는 YOLO 모델을 초기화하는 함수.

    # Run app: 앱 실행
    app.run(host='0.0.0.0', port=port, debug=False) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice) : debug = True를 사용하지 마세요. 모델이 두번 로드됩니다. stackoverflow의 질문과 답 링크
    # 호스트 0.0.0.0 / 포트 위에서 설정한 port