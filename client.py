
'''
   ____   _   _                  _                       
  / ___| | | (_)   ___   _ __   | |_       _ __    _   _ 
 | |     | | | |  / _ \ | '_ \  | __|     | '_ \  | | | |
 | |___  | | | | |  __/ | | | | | |_   _  | |_) | | |_| |
  \____| |_| |_|  \___| |_| |_|  \__| (_) | .__/   \__, |
                                          |_|      |___/ 
                                          
The following lines of code show how to make requests to the API

Flask를 사용하여 만들어진 API에 대한 요청을 사용하는 예제
'''

import requests
# Python에서 HTTP 요청을 보내는 기능을 제공하는 외부 라이브러리인 requests를 가져오는 코드

# ====================== Public image ====================== #
#여기 밑에 3개는 이미지

# Saving txt file : &save_txt=T는 쿼리 매개변수로서, 서버에게 이미지를 처리한 결과를 텍스트 파일로 저장하도록 지시. 즉, 해당 이미지에 대한 객체 감지 결과가 텍스트 파일에 저장됨.
#해당 이미지에 대한 객체 감지 결과가 텍스트 파일에 저장됨.
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg&save_txt=T',
                    verify=False)
print(resp.content) #결과가 resp.content에 저장

# Without save txt file, just labeling the image : Zidane의 이미지를 가져와서 레이블만 달린 이미지를 얻기 위해 API에 요청함. 
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg',
                    verify=False)
print(resp.content) #결과가 resp.content에 저장

# You can also copy and paste the following url in your browser : 다음 URL을 복사하여 브라우저에 붙여넣을 수 있음.
#주어진 URL을 복사하여 웹 브라우저의 주소 표신줄에 붙여넣으면 접근 가능.
'http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg'


# ====================== Public video ====================== #
#여기 아래 두개는 get방식 비디오

# (Youtube or any public server). It is not ready (yet) to return all frames labeled while using save_txt=T. So, don't try it!

resp = requests.get('http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI',
                    verify=False)

# You can also copy and paste the following url in your browser
'http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI'

# ====================== Send local file ==================== #
#위는 get 방식

#여기는 post 방식
url = 'http://0.0.0.0:5000/predict'
file_path = 'data/images/bus.jpg'

params = {
    'save_txt': 'T'
}

with open(file_path, "rb") as f:
    response = requests.post(url, files={"myfile": f}, data=params, verify=False)

print(response.content)