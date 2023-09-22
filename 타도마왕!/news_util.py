from selenium.webdriver.common.by import By
# 여기까지를 함수로 만들기
def article_value(dirver,href):
    chartDiv=driver.find_element(By.CLASS_NAME,'u_cbox_chart_cont')
    innerChart=chartDiv.find_element(By.CLASS_NAME,'u_cbox_chart_cont_inner')
    perNum=innerChart.find_elements(By.CLASS_NAME,'u_cbox_chart_per')
    names=innerChart.find_elements(By.CLASS_NAME,'u_cbox_chart_cnt')
    
    result_dict = {}
    
    for i in range(min(len(perNum), len(names))):
        key = names[i].text
        value = perNum[i].text
        result_dict[key] = value
        
    return result_dict  # result_dict 반환
        
     
        
# # # 여기까지를 하나의 함수로 만들기
def ranking_news(driver, href):
    thumb_list = driver.find_elements(By.CLASS_NAME, 'as_thumb')
    a_tags = [thumb.find_element(By.TAG_NAME, 'a') for thumb in thumb_list]

    Id_dic = {}
    
    for a_tag in a_tags:
        href = a_tag.get('href')

        result_dict = article_value(driver, href)  # article_value 함수 호출

        split = href.split('?')
        split = split[0].split('/')
        Id = split[-1]

        for key, value in result_dict.items():
            Id_dic[key] = value
        
    return Id_dic
