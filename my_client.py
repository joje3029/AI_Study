import requests

# url = 'http://0.0.0.0:5001/predict' #url 주소 원래 5000인데 일부러 5001로 함
file_path = 'data/images/bus.jpg' #파일 경로 : 버스 이미지.

url = 'http://127.0.0.1:5000/predict'

#파라미터를 정의하는 부분

params = {
    'save_txt': 'T'  #save_txt라는 파라미터를 T로 설정하고 있음. API에 대한 요청 시에 사용
                    #현재 텍스트 파일로 저장하겠다는 의미로 사용중
}

#파일패스 경로
with open(file_path, "rb") as f:                        #앞에서 정의한 파라미터 요청에 추가 : save_txt 파라미터가 전송
    response = requests.post(url, files={"myfile": f}, data=params, verify=False) # SSL 인증서의 유효성 검사를 비활성화하고 있음. 
    #POST로 요청을 보내고 있음.  파일 첨부(myfile이라는 키로 파일을 서버에 전송)

print(response.content)

# 즉, 파일 경로의 버스이미지에는 사람 2에 버스니까 coco모델이 학습했으면 결과가 객체로 나와야하잖아.
