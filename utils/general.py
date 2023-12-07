# Utils by Henry Navarro : Henry Navarro에 의한 유틸리티

"""
This module contains utility functions and constants for various purposes.
이 모듈은 다양한 목적을 위한 유틸리티 함수와 상수를 포함하고 있음.

"""

import os
from pathlib import Path #pathlib의 Path 사용

import json # Henry



FILE = Path(__file__).resolve() # Path의 파일경로를 절대 경로롤 바꿔서 FILE에 넣음
# resolve() : 해당 경로는 절대 경로로 변경
ROOT = FILE.parents[1]  # YOLOv8API root directory
RANK = int(os.getenv('RANK', -1)) # RANK 환경 변수의 값을 가져오는데, 만약 RANK가 설정되어 있지 않으면 기본값으로 -1을 사용함.=> RANK가 설정되어있지 않을 때의 기본 동작을 정의할 수 있음.
    
def update_options(request):
    """
    Args:
    # 인수
    - request: Flask request object
    # 플라스크의 request 객체
    
    Returns:
    # 반환값
    - source: URL string
    # URL 문자열
    - save_txt: Boolean indicating whether to save text or not
    # save_txt : 텍스트를 저장할지 여부를 나타내는 부울값
    """
    
    # GET parameters : get 일 때
    if request.method == 'GET':
        #all_args = request.args # TODO: get all parameters in one line : 모든 매개변수를 한 줄로 가져오기
        source = request.args.get('source') # request에서 source라는 이름을 꺼내서 source 변수에 담음
        save_txt = request.args.get('save_txt') # request에서 save_txt라는 이름을 꺼내서 save_txt 변수에 담음

    
    # POST parameters : post 일 때
    if request.method == 'POST':
        json_data = request.get_json() #Get the POSTed json : post라서 get_json으로 먼저 요청에서 가져오는거구나.
        json_data = json.dumps(json_data) # API receive a dictionary, so I have to do this to convert to string : API는 딕셔너리를 받기 때문에 이를 문자열로 변환해야 합니다.
        dict_data = json.loads(json_data) # Convert json to dictionary  : JSON을 딕셔너리로 변환합니다.
        source = dict_data['source'] # 딕셔너리로 변형한거에서 key가 source인걸 찾아서 source에 담음. 
        save_txt = dict_data.get('save_txt', None) #딕셔너리로 변형한거에서 key가 save_txt 인걸 찾아서 save_txt에 담음. key가 save_txt일때 디폴트 값으로 None을 사용하겠다.
    
    return source, bool(save_txt) # get 이든  post이든  if로 변형 다하고 난거 return. 
