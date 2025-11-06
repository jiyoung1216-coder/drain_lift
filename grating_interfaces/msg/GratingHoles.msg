# 시간 스탬프: 데이터가 생성된 시간을 기록
builtin_interfaces/Time stamp

# 빗물받이 구멍 검출 정보
uint8 num_holes           # 실제로 찾은 구멍 개수 (0~2)

# 정규화된 이미지 좌표 (u = x/width, v = y/height)
# u1, v1, u2, v2 순서로, 구멍이 없으면 0.0으로 채움
float32[4] holes_uv       

# 데이터가 측정된 프레임 ID (예: 카메라 이름)
string frame_id