import json
from openai import OpenAI

# 육하원칙 생성 함수 (gpt 사용)
def get_llm_response(promt, news):
  client = OpenAI(api_key="api키 입력하면 됨")
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": promt},
      {"role": "user", "content": news}
    ]
  )

  return response.choices[0].message.content


def main(original_news):
  promt = '''데이터 설명

1. 야구기사를 육하원칙으로 정리한 데이터이다.

요구 사항

1. 육하원칙을 살펴본 후 하나의 뉴스레터로 만들어라
2. 실제 뉴스레터 형식으로 만들어라 (친근한 말투)
3. 한국어로 작성해라
4. 맨 첫문장은 뉴스레터 제목으로 사용해라
5. '첫 인사말'과 '마무리 인사말'은 생략해라 (안녕하세요 같은 인사말 제외)
6. "야구 팬 여러분" 과 같은 시청자를 지칭하는 단어는 생략해라
7. 주관적인 의견은 제외 및 생성하지 말고 객관적인 정보만 참고해서 뉴스레터 생성해라
8. 700로 작성해라.
9. 만약 감독과 선수의 인터뷰 내용이 '없음'이면 인터뷰 관련 내용을 작성하지 마라. '인터뷰는 없었습니다' 느낌의 문장도 생성하지 마라. 
10. 대제목은 사용하되 소제목은 사용하지 마라'''

  llm_response = get_llm_response(promt, original_news)

  return llm_response


if __name__ == '__main__':
  pass