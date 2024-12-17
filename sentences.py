import pandas as pd
import numpy as np

import chardet

# 파일의 인코딩 확인
with open('data/jeju_data.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result)

# 감지된 인코딩으로 파일 읽기
df = pd.read_csv('data/jeju_data.csv', encoding=result['encoding'])

# 자연어 문장 생성 함수
def generate_sentence(row):

    # 업종 및 주소 처리
    sentence = f"{row['가맹점명']}은(는) {row['업종']}을(를) 판매하고 있으며, {row['행정구역']}에 위치하고 있습니다. "
    
    sentence += f"상세 주소는 {row['주소']} 입니다. "

    # 이용건수구간 처리
    if (row['이용건수구간']) == 1:
        sentence += f"이용건수구간은 1구간으로, "
    elif (row['이용건수구간']) == 2:
        sentence += f"이용건수구간은 2구간으로, "
    elif (row['이용건수구간']) == 3:
        sentence += f"이용건수구간은 3구간으로, "
    elif (row['이용건수구간']) == 4:
        sentence += f"이용건수구간은 4구간으로, "
    elif (row['이용건수구간']) == 5:
        sentence += f"이용건수구간은 5구간으로, "
    else:
        sentence += f"이용건수구간은 6구간으로, "

    if (row['이용건수구간']) == 1:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 90% 입니다. "
        # sentence += f"이용객이 적어, 방문하신다면 여유롭게 이용하실 수 있습니다. "
    elif (row['이용건수구간']) == 2:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 75%~90% 입니다. "
        # sentence += f"이용객이 적어, 방문하신다면 여유롭게 이용하실 수 있습니다. "
    elif (row['이용건수구간']) == 3:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 50%~75% 입니다. "
        # sentence += f"시간과 요일을 잘 선택하여 방문하신다면 여유롭게 이용하실 수 있습니다. "
    elif (row['이용건수구간']) == 4:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 25%~50% 입니다. "
        # sentence += f"시간과 요일을 잘 선택하여 방문하신다면 여유롭게 이용하실 수 있습니다. "
    elif (row['이용건수구간']) == 5:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 10%~25% 입니다. "
        # sentence += f"이용객이 많아 혼잡할 수 있습니다. "
    else:
        sentence += f"제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 10% 입니다. "
        # sentence += f"이용객이 많아 혼잡할 수 있습니다. "
    
    # 이용금액구간 처리
    if (row['이용금액구간']) == 1:
        sentence += f"이용금액구간은 1구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 90% 입니다. "
    elif (row['이용금액구간']) == 2:
        sentence += f"이용금액구간은 2구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 75%~90% 입니다. "
    elif (row['이용금액구간']) == 3:
        sentence += f"이용금액구간은 3구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 50%~75% 입니다. "
    elif (row['이용금액구간']) == 4:
        sentence += f"이용금액구간은 4구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 25%~50% 입니다. "
    elif (row['이용금액구간']) == 5:
        sentence += f"이용금액구간은 5구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 10%~25% 입니다. "
    else:
        sentence += f"이용금액구간은 6구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 10% 입니다. "


    # 요일별 이용건수 처리
    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    non_zero_weekdays_slot = [slot for slot in weekdays if row[slot] != 0]
    max_day = max(non_zero_weekdays_slot, key=lambda day: row[day])
    min_day = min(non_zero_weekdays_slot, key=lambda day: row[day])
    sentence += f"요일별 이용객 비중은 월요일 {row['월요일']}%, 화요일 {row['화요일']}%, 수요일 {row['수요일']}%, 목요일 {row['목요일']}%, 금요일 {row['금요일']}%, 토요일 {row['토요일']}%, 일요일 {row['일요일']}% 이며, "

    sentence += f"가장 많은 사람들이 방문하는 요일은 {max_day}이고, 가장 적은 사람들이 방문하는 요일은 {min_day}입니다. "

    # 시간대별 이용건수 처리
    time_slots = ['5시11시(아침)', '12시13시(점심)', '14시17시(오후)', '18시22시(저녁)', '23시4시(새벽)']
    non_zero_time_slot = [slot for slot in time_slots if row[slot] != 0]
    max_slot = max(non_zero_time_slot, key=lambda slot: row[slot])
    min_slot = min(non_zero_time_slot, key=lambda slot: row[slot])
    sentence += f"시간대별 이용객 비중은 5시~11시(아침) {row['5시11시(아침)']}%, 12시~13시(점심) {row['12시13시(점심)']}%, 14시~17시(오후) {row['14시17시(오후)']}%, 18시~22시(저녁) {row['18시22시(저녁)']}%, 23시~4시(새벽) {row['23시4시(새벽)']}% 입니다. "
    
     # 시간대별 이용건수 중 최대값/최소값/두 번째 최소값 찾기
    def format_time_slot(slot):
        start = slot[:slot.index('시') + 1]  # '시'까지의 부분
        end = slot[slot.index('시') + 1:]  # '시' 이후 부분
        return start + '~' + end

    max_slot_formatted = format_time_slot(max_slot)
    min_slot_formatted = format_time_slot(min_slot)
       
    sentence += f"{max_slot_formatted} 사이에 이용객의 수가 가장 많은 시간대이며, {min_slot_formatted} 사이에 이용객의 수가 가장 적은 시간대입니다. "
    
    # 성별 정보 처리
    def compare_gender_usage(male_count, female_count):
        if male_count > female_count:
            return "남성"
        elif female_count > male_count:
            return "여성"
        else:
            return "동일"
        
    total_gender = row['남성'] + row['여성']
    dominant_gender = compare_gender_usage(row['남성'], row['여성'])

    if dominant_gender == "동일":
        sentence += f"이용객 성별 비율은 남성 {row['남성']/total_gender:.2%}, 여성 {row['여성']/total_gender:.2%}으로, 성별 상관없이 방문합니다. "
    else:
        sentence += f"이용객 성별 비율은 남성 {row['남성']/total_gender:.2%}, 여성 {row['여성']/total_gender:.2%}으로, {dominant_gender} 고객들이 더 많이 방문합니다. "


    # 연령대 정보 처리
    age_groups = ['20대이하', '30대', '40대', '50대', '60대이상']
    max_age_group = max(age_groups, key=lambda age: row[age])
    min_age_group = min(age_groups, key=lambda age: row[age])

    sentence += f"연령대별 이용객 비중은 20대 이하 {row['20대이하']}%, 30대 {row['30대']}%, 40대 {row['40대']}%, 50대 {row['50대']}%, 60대 이상 {row['60대이상']}% 입니다. "
    sentence += f"{max_age_group} 이용객이 가장 많이 방문하였으며, {min_age_group} 이용객이 가장 적게 방문하였습니다. "

    # 현지인 정보 처리
    sentence += f"현지인 이용 건수 비중은 {row['현지인이용건수비중']}% 이므로, "
    if round(row['현지인이용건수비중'],2) >= 50:
        sentence += f"현지인 이용 비중이 관광객 이용 비중보다 높습니다. "
    sentence += f"관광객 이용 비중이 현지인 이용 비중보다 높습니다. "

    # url 처리
    url = str(row['URL'])
    # URL이 null인지 체크하여 처리
    if pd.isna(url) or url == 'nan':
        sentence += "URL이 없습니다."
    else:
        sentence += f"URL은 {url} 입니다."
    
    return sentence

# 새로운 'sentence' 칼럼 추가
df['sentence'] = df.apply(generate_sentence, axis=1)

# 결과를 새 CSV 파일로 저장
df.to_csv('jeju_sentences.csv', index=False, encoding='utf-8-sig') 

print("처리가 완료되었습니다. 결과는 'jeju_sentences.csv' 파일에 저장되었습니다.")