import pandas as pd
import numpy as np
import chardet

# 파일의 인코딩 확인
with open('data/jeju_sentences.csv', 'rb') as f:
    result = chardet.detect(f.read())

# 감지된 인코딩으로 파일 읽기
df = pd.read_csv('data/jeju_sentences.csv', encoding=result['encoding'])

# 각 섹션별 문장 생성 함수
def generate_sections(row):
    # 업종명 처리
    section1 = f"업종명 {row['가맹점명']}은(는) {row['업종']}을(를) 판매하고 있습니다."

    # 주소 처리
    section2 = f"주소는 {row['행정구역']}에 위치하고 있습니다. 상세 주소는 {row['주소']} 입니다."

    # 이용건수구간 처리
    usage_count_section = {
        1: "1구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 90% 입니다. ",
        2: "2구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 75%~90% 입니다. ",
        3: "3구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 50%~75% 입니다. ",
        4: "4구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 25%~50% 입니다. ",
        5: "5구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 10%~25% 입니다. ",
        6: "6구간으로, 제주도 방문객의 전체 이용 건수 중 이용 비중이 하위 10% 입니다. "
    }
    section4 = f"이용건수구간은 {usage_count_section[row['이용건수구간']]}"

    # 이용금액구간 처리
    usage_amount_section = {
        1: "1구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 90% 입니다.",
        2: "2구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 75%~90% 입니다.",
        3: "3구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 50%~75% 입니다.",
        4: "4구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 25%~50% 입니다.",
        5: "5구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 10%~25% 입니다.",
        6: "6구간으로, 제주도 방문객의 전체 이용 금액 중 이용 금액이 하위 10% 입니다."
    }
    section5 = f"이용금액구간은 {usage_amount_section[row['이용금액구간']]}"

    # 요일별 이용건수 처리
    weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    max_day = max(weekdays, key=lambda day: row[day])
    min_day = min(weekdays, key=lambda day: row[day])
    section6 = (f"요일별 이용객 비중은 월요일 {row['월요일']}%, 화요일 {row['화요일']}%, 수요일 {row['수요일']}%, "
                f"목요일 {row['목요일']}%, 금요일 {row['금요일']}%, 토요일 {row['토요일']}%, 일요일 {row['일요일']}% 이며, "
                f"가장 많은 사람들이 방문하는 요일은 {max_day}이고, 가장 적은 사람들이 방문하는 요일은 {min_day}입니다.")

    # 시간대별 이용건수 처리
    time_slots = ['5시11시(아침)', '12시13시(점심)', '14시17시(오후)', '18시22시(저녁)', '23시4시(새벽)']
    max_slot = max(time_slots, key=lambda slot: row[slot])
    min_slot = min(time_slots, key=lambda slot: row[slot])
    
    def format_time_slot(slot):
        start = slot[:slot.index('시') + 1]
        end = slot[slot.index('시') + 1:]
        return start + '~' + end

    section7 = (f"시간대별 이용객 비중은 아침 {row['5시11시(아침)']}%, 점심 {row['12시13시(점심)']}%, 오후 {row['14시17시(오후)']}%, "
                f"저녁 {row['18시22시(저녁)']}%, 새벽 {row['23시4시(새벽)']}% 입니다. "
                f"{format_time_slot(max_slot)} 사이에 이용객의 수가 가장 많은 시간대이며, {format_time_slot(min_slot)} 사이에 이용객의 수가 가장 적은 시간대입니다. ")

    # 성별 정보 처리
    total_gender = row['남성'] + row['여성']
    
    def compare_gender_usage(male_count, female_count):
        if male_count > female_count:
            return "남성"
        elif female_count > male_count:
            return "여성"
        else:
            return "동일"
        
    dominant_gender = compare_gender_usage(row['남성'], row['여성'])
    
    if dominant_gender == "동일":
        section8 = (f"이용객 성별 비율은 남성 {row['남성']/total_gender:.2%}, 여성 {row['여성']/total_gender:.2%}으로, "
                    f"성별 상관없이 방문합니다. ")
    else:
        section8 = (f"이용객 성별 비율은 남성 {row['남성']/total_gender:.2%}, 여성 {row['여성']/total_gender:.2%}으로, "
                    f"{dominant_gender} 고객들이 더 많이 방문합니다. ")

    # 연령대 정보 처리
    age_groups = ['20대이하', '30대', '40대', '50대', '60대이상']
    max_age_group = max(age_groups, key=lambda age: row[age])
    min_age_group = min(age_groups, key=lambda age: row[age])
    
    section9 = (f"연령대별 이용객 비중은 20대 이하 {row['20대이하']}%, 30대 {row['30대']}%, "
                f"40대 {row['40대']}%, 50대 {row['50대']}%, 60대 이상 {row['60대이상']}% 입니다. "
                f"{max_age_group}가 가장 많이 방문하며, {min_age_group}가 가장 적게 방문합니다. ")

    # 현지인 정보 처리
    local_usage_ratio = round(row['현지인이용건수비중'],2)
    
    section10 = (f"현지인 이용 건수 비중은 {local_usage_ratio}% 이므로, "
                 f"{'현지인' if local_usage_ratio >= 50 else '관광객'} 사용 비중이 높습니다.")

    # URL 처리
    url = str(row.get('URL', ''))
    
    if pd.isna(url) or url == 'nan':
        section11 = "URL이 없습니다."
    else:
        section11 = f"URL은 {url} 입니다."

    return {
        '업종명': section1,
        '주소': section2,
        '이용건수구간': section4,
        '이용금액구간': section5,
        '요일별': section6,
        '시간대별': section7,
        '성별': section8,
        '연령별': section9,
        '현지인': section10,
        'url': section11
    }

# 각 섹션을 별도의 칼럼에 추가
sections_df = df.apply(generate_sections, axis=1).apply(pd.Series)

# 원래 데이터프레임과 병합하여 저장
result_df = pd.concat([df, sections_df], axis=1)

# 결과를 새 CSV 파일로 저장
result_df.to_csv('alphastorm_data.csv', index=False, encoding='utf-8-sig')

print("처리가 완료되었습니다. 결과는 'alphastorm_data.csv' 파일에 저장되었습니다.")