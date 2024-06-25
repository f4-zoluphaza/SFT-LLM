import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

article_text = """키움 히어로즈가 한화 이글스와의 3연전을 쓸어 담고 7연승을 내달렸다.
키움은 7일 서울 고척스카이돔에서 열린 2024 신한 SOL 뱅크 KBO리그 홈경기에서 연장 접전 끝에 한화를 4-3으로 제압했다. 이로써 키움은 개막 4연패 이후 쾌조의 7연승을 달리며 리그 상위권으로 도약했다.
반면 한화는 개막 후 10경기까지 구단 사상 최고 승률(8승 2패)을 찍었다가 이후 3연패로 분위기가 가라앉았다.
승부는 3-3으로 맞선 연장 11회말, 키움 주장 김혜성이 선두타자로 나서 답답한 경기에 마침표를 찍었다. 김혜성은 한화 투수 이태양을 상대로 풀카운트에서 6구째 직구를 잡아 당겨 오른쪽 담장 너머로 보냈다. 김혜성의 시즌 4호 홈런으로 KBO리그 데뷔한 이래 처음 맛본 끝내기 홈런이다.
양 팀 선발투수 김민우(한화), 김선기(키움)는 1회부터 차례로 솔로포를 맞았다. 한화 문현빈이 프로 데뷔 후 처음으로 선두타자 홈런을 쏘아 올렸고, 키움 김혜성은 2사 후 우월 솔로 아치를 그렸다.
이후 경기는 투수전 양상으로 흘렀다. 김선기가 마운드에서 내려간 6회초, 한화가 먼저 앞서가는 점수를 올렸다.
키움 두 번째 투수 김연주는 볼넷 2개로 2사 1, 3루 위기를 자초했고 폭투를 던져 역전을 허용했다. 이어 이도윤에게 우전 적시타를 맞아 한 점을 더 잃었다.
6회까지 김민우에게 3안타로 꽁꽁 묶여있던 키움은 7회말 송성문의 투런포로 경기를 단번에 원점으로 돌렸다.
한화 유격수 이도윤은 1사 후 이형종의 땅볼 때 실책성 송구를 던져 타자 주자를 살려 보냈고, 송성문이 우월 동점포를 작렬했다.
역투를 이어가던 김민우는 결국 7이닝 91구 5피안타(2홈런) 7탈삼진 3실점(3자책)을 기록, 승리요건을 채우지 못하고 마운드에서 내려왔다.
양 팀은 정규 이닝 안에 승부를 가리지 못하고 연장전으로 향했다.
두 팀은 체력전 속에서도 집중력을 잃지 않고 차례로 호수비를 선보였다. 키움 중견수 이주형은 10회초 2사 1, 3루에서 우중간 외야로 날아가는 채은성의 안타성 타구를 외야 펜스에 부딪혀가며 잡아냈다.
10회말에는 한화 2루수 문현빈의 중계 플레이가 돋보였다. 이주형이 1사 주자 없는 상황에서 2루타를 치고 3루까지 달리자 문현빈은 중견수에게서 공을 받아 3루에 빠르게 뿌려 아웃을 잡았다.
한편 한화는 이날 득점 찬스를 살리지 못하고 잔루를 15개 쏟아냈다. 한화는 4회 2사 1, 3루와 6회 2사 만루, 9사 2사 만루 기회를 모두 날렸다. 노시환은 이날 6타수 4안타로 활약했으나 한 번도 홈 플레이트를 밟지 못했다.
결국 승리는 마지막 집중력에서 앞선 키움의 몫이었다. 
홍성욱 기자  mark@thesportstimes.co.kr
<저작권자 © 스포츠타임스, 무단 전재 및 재배포 금지>
"""

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_ids = tokenizer(
    [WHITESPACE_HANDLER(article_text)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=84,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
