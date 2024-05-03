# JSONL 파일 경로
file_path = "./test.jsonl"

# 파일을 읽어서 줄 바꿈 문자를 제거한 후에 다시 파일에 쓰는 함수
def remove_newlines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 줄 바꿈 문자를 제거하고 새로운 리스트에 저장
    lines = [line.strip() for line in lines]

    # 파일을 쓰기 모드로 열어서 새로운 리스트를 파일에 씀
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line)

# 함수 호출하여 줄 바꿈 문자 제거
remove_newlines(file_path)