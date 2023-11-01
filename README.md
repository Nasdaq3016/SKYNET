우선순위로 DFX 기반 하드웨어를 만드는 것을 1순위로 함

각 파트별로 메인 팀원을 선정하고 나머지는 해당 부분에 서브로 들어가는 형식으로 진행할 것

컴퓨팅 코어 부분에서는 (이윤태)
INT8 기반 MPU, LayerNorm, Mask, Gelu, Softmax, ReduceMAx, Residual 유닛을 Custom HW로 만드는 것을 목표로 함
그리고 해당 유닛을 다음과 같은 구조로 엮는 것을 목표로 하는 HW를 구현하는 것을 목표로 함, TB 검증 철저히 실행할 것
Llama2 묘사와 같이 진행할 것
![image](https://github.com/Nasdaq3016/SKYNET/assets/108527148/b7b46374-02d5-401f-b1cd-72a5a74d6ec4)


Llama2 묘사 (조영민)
1. 해당 깃허브에서 Llama2의 Sudo Code를 확인 가능
2. Estimater의 HW Modeling 혹은 CNN_ff를 참고하여 구현하고자 하는 모듈들의 기능을 정의하고 일종의 명령어 집합을 만들 수 있음
3. 그리고 해당 연산에 필요한 Dummy Data를 만들 수 있음 (TB와 같이 사용할 것)

Off-Chip Part (박종혁)
해당 파트에 대해서 각 루프핑 테스트를 진행할 것
AXI와 PCIE에 대한 프로토콜 이해를 선행하고, 설계도를 도식화 할 것
1. PCIE : QDMA IP
2. HBM : HBM IP
3. DDR : DDR4 SDRAM IP

위의 부분을 CES 2024 전까지 끝낼 것.

연구 파트.
AWQ 개선 연구는 좀 더 고민이 필요할 것으로 판단, 현재 보류 상태
SpecInfer의 병렬 Decoder 연산을 하드웨어 적으로 가속 가능할 것으로 예상, 특히 CPU로 오가는 부분을 End-End로 만들 수 있을 것이라 판단.

2024년 3월까지 연구 주제를 선정 및 실험을 진행할 것.
