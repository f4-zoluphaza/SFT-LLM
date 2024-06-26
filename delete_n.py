# JSONL 파일 경로
file_path = "./test.jsonl"

# 파일을 읽어서 큰따옴표를 작은따옴표로 바꾸고 다시 파일에 쓰는 함수
def process_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 큰따옴표를 작은따옴표로 바꾼 후 새로운 리스트에 저장
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.replace('"', "'")
        cleaned_lines.append(cleaned_line)

    # 파일을 쓰기 모드로 열어서 새로운 리스트를 파일에 씀
    with open(file_path, 'w') as file:
        for line in cleaned_lines:
            file.write(line)

# 파일을 읽어서 줄 바꿈 문자를 띄어쓰기로 변경한 후에 다시 파일에 쓰는 함수
def replace_newlines_with_spaces(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 줄 바꿈 문자를 띄어쓰기로 바꾸고 새로운 리스트에 저장
    cleaned_content = ' '.join([line.strip() for line in lines])

    # 파일을 쓰기 모드로 열어서 새로운 리스트를 파일에 씀
    with open(file_path, 'w') as file:
        file.write(cleaned_content)

# 함수 호출하여 큰따옴표를 작은따옴표로 변경
process_jsonl_file(file_path)

# 함수 호출하여 줄 바꿈 문자를 띄어쓰기로 변경
replace_newlines_with_spaces(file_path)
