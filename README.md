# Mass-Spring-Cloth-Simulation

<video alt="Simulation Video" src="https://github.com/user-attachments/assets/ad55bd22-e247-4d9a-9e49-7633f3611f2b"></video>



## 몇가지 질문들과 해답



### 1. 왜 dashpot_damping 계수에 quad_size를 곱해줘야할까?
    
  ```python
  force += -v_ij.dot(d) * (dashpot_damping * quad_size) * d # dashpot damping force
  ```
  
  $F_{new} \leftarrow F_{old} - (v_{ij} \cdot d ) \cdot (\frac{1}{n} \cdot k_{damp}) \cdot d$
  
  n이 2배 증가하면(옷 입자의 수가 2배 증가),
  
  입자 사이사이에 똑같은 크기의 힘이 추가로 두배씩 생김 → 옷의 복원력이 두배 강해짐
  
  입자의 개수가 2배 증가했다면, damping 계수를 1/2배로 줄여서
  
  입자의 개수와 무관하게 옷의 성질이 유지되도록 해야함

---

### 2. 왜 drag_damping 을 복잡하게 지수형태로 쓸까?
    
  ```python
  for i in ti.grouped(x):
          v[i] *= ti.exp(-drag_damping * dt) # drag damping independent of dt
  ```
  
  $v_{new} \leftarrow v_{old} \cdot e^{-k_{drag} \cdot \Delta t}$
  
  물론 $e^{-k_{drag} \cdot \Delta t}$가 통쨰로 상수이기에 $v_{new} \leftarrow v_{old} \cdot k$ 이렇게 해도 되지만,
  
  - 의미를 명확히 하기 위함
  - dt와 시뮬레이션 결과의 독립성을 유지하기 위함
  
  - $e$를 쓰는 이유
      - 공기저항 식
          
          $F_{drag} = - k \cdot v$
          
          $\frac{dv}{dt} = - \frac{1}{m} \cdot k \cdot v$ : 속도의 변화율이 현재 속도에 비례한다.
          
          변수분리법으로 정리해보면,
          
          $\frac{1}{v} dv = (- \frac{1}{m} k) dt$
          
          적분하면, $v = v_0 \cdot e^{- \frac{1}{m} kt}$
          
      
      따라서 시뮬레이션에서는 $v_{new} = v_{old} \cdot e^{ - \frac{1}{m} k\Delta t}$ (이 식에서는 $m=1$로 가정한 것)
      
  
  - $e^{-k_{drag} \cdot \Delta t}$는 통째로 상수인데, 왜 $e^{-k_{drag} \cdot \Delta t} = k$ 이렇게 간단히 표현하지 않을까?
      - $v_{new} \leftarrow v_{old} \cdot k$ 이렇게 놓았을때, dt가 변한다면,
          - 상황1 (dt = 0.01): 1초 동안 k가 100번 곱해짐
              
              → 1초동안 $k^{100}$ 만큼 감소
              
          - 상황2 (dt = 0.001) : 1초 동안 k가 1000번 곱해짐
              
              → 1초동안 $k^{1000}$ 만큼 감소
              
          
          정밀도를 높이기 위해 dt를 작게 했더니, 속도가 같은 시간동안 훨씬 더 많이 감소하게 됨.
          (damping 효과 증폭)
          
      - $v_{new} \leftarrow v_{old} \cdot e^{-k_{drag} \cdot \Delta t}$ 이렇게 놓았을때, dt가 변한다면,
          - 상황1 (dt = 0.01): 1초 동안 $e^{-k_{drag} \cdot 0.01}$이 100번 곱해짐
              
              → 1초동안 $(e^{-k_{drag} \cdot 0.01})^{100} = e^{-k_{drag}}$ 만큼 감소
              
          - 상황2 (dt = 0.001) : 1초 동안
              
              → 1초동안 $(e^{-k_{drag} \cdot 0.001})^{1000} = e^{-k_{drag}}$ 만큼 감소
              
          
          정밀도를 높여도 초당 속도 감소량은 일정함.
            
---

### 3. `substep()`함수에서 왜 for문을 중간중간 끊어야하나?
    
  ```python
  @ti.kernel
  def substep():
      for i in ti.grouped(x):
          v[i] += gravity * dt
      
      for i in ti.grouped(x):
          force = ti.Vector([0.0, 0.0, 0.0])
          for spring_offset in ti.static(spring_offsets):
              j = i + spring_offset
  
              if 0 <= j[0] < n and 0 <= j[1] < n:
                  x_ij = x[i] - x[j]
                  v_ij = v[i] - v[j]
  
                  d = x_ij.normalized() # direction from j to i
                  current_dist = x_ij.norm() # current distance between i and j
                  original_dist = quad_size * float(spring_offset).norm() # original distance between i and j
  
                  force += -spring_Y[None] * (current_dist / original_dist - 1) * d # spring force
                  force += -v_ij.dot(d) * (dashpot_damping[None] * quad_size) * d # dashpot damping force
  
          v[i] += force * dt # mass = 1.0
  
      for i in ti.grouped(x):
          v[i] *= ti.exp(-drag_damping[None] * dt) # drag damping independent of dt
  
          # Collision with the ball
          offset_to_center = x[i] - ball_center[0]
          if offset_to_center.norm() <= ball_radius:
              normal = offset_to_center.normalized()
              v_normal = v[i].dot(normal)
  
              v[i] -= (1 + restitution[None]) * min(v_normal, 0) * normal # restitution
              v[i] -= friction[None] * (v[i] - v_normal * normal) # friction
              x[i] = ball_center[0] + ball_radius * normal # position projection
  
          x[i] += dt * v[i]
  ```
  
  - `for spring_offset in ti.static(spring_offsets):` ← 이 반복문에서 문제 발생함
      - 이 반복문에서는 다른 입자의 위치와 속도를 참고하여 연산을 진행한다.
      - 한 스레드에서는 최종 입자 위치 계산까지 완료했지만,
      다른 스레드에서는 이제 첫번째 줄을 실행 중일 수도 있다.
      - 이 미세한 균열이 중첩되면 시뮬레이션이 폭발한다.
      - 따라서 다른 입자의 상태를 참고하는 병렬처리 후에는 꼭 for문을 종료하여 모든 입자의 계산 완료를 보장해야함.
  - 병렬처리 순서
      - 중력 모든 입자에 적용
      - 그 위치에서 천의 복원력 적용(spring force) 및 속도 계산
      - 계산한 속도에서 drag, friction, restitution 각각 적용
