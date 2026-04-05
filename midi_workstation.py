import pygame
import mido
import numpy as np
import sys
import random

# 1. 초기화 및 오디오 설정
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
info = pygame.display.Info()
SW, SH = info.current_w, info.current_h
screen = pygame.display.set_mode((SW, SH), pygame.FULLSCREEN)
pygame.display.set_caption("Advanced MIDI Workstation - Pro Visual Edition")

# 폰트 및 색상 설정
font = pygame.font.SysFont("AppleGothic", 16)
small_font = pygame.font.SysFont("AppleGothic", 11, bold=True)
large_font = pygame.font.SysFont("AppleGothic", 22, bold=True)
tempo_font = pygame.font.SysFont("Courier", 32, bold=True)
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (50, 50, 50)
GREEN, POINT_COLOR = (0, 255, 0), (0, 200, 255)
EXIT_RED, PLAY_GREEN = (200, 50, 50), (50, 200, 50)

# 2. 상태 변수 및 데이터
is_playing = False
tempo = 120
current_step = 0
last_tick = pygame.time.get_ticks()
octave_offset = 0
synth_octave = 0
bass_octave = 0
active_notes = set()

# 드럼 데이터
DRUM_NAMES = ["CRASH2", "CRASH1", "TOM4", "TOM3", "TOM2", "TOM1", "HH", "SD", "BD"]
drum_grid = np.zeros((9, 16), dtype=int)
drum_pitches = [0] * 9

# 디스코 리듬 초기화
for s in [0, 4, 8, 12]: drum_grid[8, s] = 1
for s in [4, 12]: drum_grid[7, s] = 1
for s in range(0, 16, 2): drum_grid[6, s] = 1

bass_grid = np.zeros((12, 16), dtype=int)
synth_grid = np.zeros((12, 16), dtype=int)

MIXER_GAIN = {"DRUM": 0.7, "BASS": 0.6, "SYNTH": 0.35, "MASTER": 0.8}
OSC_TYPES = ["sine", "saw", "pulse"]

adsr_settings = {
    "BASS":  {"A": 0.02, "D": 0.2, "S": 0.5, "R": 0.2},
    "SYNTH": {"A": 0.08, "D": 0.15, "S": 0.4, "R": 0.4}
}

synth_settings = {
    "BASS": {"osc": "pulse", "lfo_freq": 2.0, "lfo_depth": 0.02, "cutoff": 800},
    "SYNTH": {"osc": "saw", "lfo_freq": 5.0, "lfo_depth": 0.03, "cutoff": 2800}
}

# --- 파형 아이콘 생성 함수 ---
def create_wave_icons():
    icons = {}
    size = (40, 12)
    for o_type in ["sine", "saw", "pulse"]:
        surf = pygame.Surface(size, pygame.SRCALPHA)
        points = []
        for x in range(size[0]):
            if o_type == "sine":
                y = size[1]//2 + int(5 * np.sin(x * 0.3))
            elif o_type == "saw":
                y = size[1] - int((x % (size[0]//2)) * (size[1] / (size[0]//2)))
            else: # pulse
                y = 2 if (x % size[0]) < size[0]//2 else size[1]-2
            points.append((x, y))
        pygame.draw.lines(surf, WHITE, False, points, 2)
        icons[o_type] = surf
    return icons

wave_icons = create_wave_icons()

# 미디 장치 설정
input_ports = mido.get_input_names()
inport = None
selected_port_name = "Select MIDI Port"
is_dropdown_open = False

# 3. 사운드 엔진 함수
def lowpass(signal, cutoff=2000, sr=44100):
    if len(signal) == 0: return signal
    rc = 1.0 / (cutoff * 2 * np.pi)
    dt = 1.0 / sr
    alpha = dt / (rc + dt)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = out[i-1] + alpha * (signal[i] - out[i-1])
    return out

def apply_adsr(t, sr, inst_name):
    conf = adsr_settings[inst_name]
    a, d, s, r = conf["A"], conf["D"], conf["S"], conf["R"]
    s_dur = 0.15
    env = np.zeros_like(t)
    a_e, d_e, s_e = int(a*sr), int((a+d)*sr), int((a+d+s_dur)*sr)
    if a_e > 0: env[:a_e] = np.linspace(0, 1, a_e)
    if d_e > a_e: env[a_e:d_e] = np.linspace(1, s, d_e - a_e)
    env[d_e:s_e] = s
    if len(env) > s_e: env[s_e:] = np.linspace(s, 0, len(env) - s_e)
    return env

def play_drum_synth(inst_idx):
    sr = 44100
    name = DRUM_NAMES[inst_idx]
    p_factor = 2 ** (drum_pitches[inst_idx] / 12.0)
    if name in ["BD", "TOM1", "TOM2", "TOM3", "TOM4"]:
        dur = 0.2 / p_factor
        t = np.linspace(0, dur, int(sr * dur), False)
        f_start = (150 if name=="BD" else 180+inst_idx*30) * p_factor
        freq_seq = np.geomspace(f_start, 40 * p_factor, len(t))
        wave = np.sin(2 * np.pi * freq_seq * t)
        env = np.exp(-t * (25 if name=="BD" else 12) * p_factor)
        vol = MIXER_GAIN["DRUM"] * MIXER_GAIN["MASTER"]
        final = (wave * env * vol * 32767).astype(np.int16)
    else:
        dur = (0.08 if name=="HH" else 0.25 if name=="SD" else 0.7) / p_factor
        t = np.linspace(0, dur, int(sr * dur), False)
        noise = np.random.uniform(-1, 1, len(t))
        env = np.exp(-t * (60 if name=="HH" else 25 if name=="SD" else 6) * p_factor)
        vol = MIXER_GAIN["DRUM"] * MIXER_GAIN["MASTER"]
        final = (noise * env * vol * 32767).astype(np.int16)
    stereo = np.column_stack((final, final))
    pygame.sndarray.make_sound(np.ascontiguousarray(stereo)).play()

def engine_bass(note_num):
    sr = 44100
    freq = 440.0 * (2.0 ** ((note_num + bass_octave - 69) / 12.0))
    conf = adsr_settings["BASS"]
    t = np.linspace(0, conf["A"]+conf["D"]+0.15+conf["R"], int(sr * (conf["A"]+conf["D"]+0.15+conf["R"])), False)
    wave = np.where((t * freq) % 1.0 < 0.5, 1.0, -1.0) # Pulse wave fix
    wave = np.tanh(wave * 2.5)
    wave = lowpass(wave, synth_settings["BASS"]["cutoff"])
    env = apply_adsr(t, sr, "BASS")
    final = (wave * env * MIXER_GAIN["BASS"] * MIXER_GAIN["MASTER"] * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((final, final))))

def engine_synth(note_num):
    sr = 44100
    freq = 440.0 * (2.0 ** ((note_num + synth_octave - 69) / 12.0))
    conf = adsr_settings["SYNTH"]
    s_conf = synth_settings["SYNTH"]
    t = np.linspace(0, conf["A"]+conf["D"]+0.15+conf["R"], int(sr * (conf["A"]+conf["D"]+0.15+conf["R"])), False)
    lfo = np.sin(2 * np.pi * s_conf["lfo_freq"] * t) * s_conf["lfo_depth"]
    mod_freq = freq * (1 + lfo)
    if s_conf["osc"] == "sine": wave = np.sin(2 * np.pi * mod_freq * t)
    elif s_conf["osc"] == "saw": wave = 2 * (t * mod_freq - np.floor(0.5 + t * mod_freq))
    else: wave = np.where((t * mod_freq) % 1.0 < 0.5, 1.0, -1.0) # Pulse wave fix
    wave = lowpass(wave, s_conf["cutoff"])
    env = apply_adsr(t, sr, "SYNTH")
    final = (wave * env * MIXER_GAIN["SYNTH"] * MIXER_GAIN["MASTER"] * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack((final, final))))

def draw_keyboard():
    kw, kh = SW // 14, 150
    by = SH - kh
    white_notes = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]
    for i, off in enumerate(white_notes):
        note = 48 + off + octave_offset
        color = POINT_COLOR if note in active_notes else WHITE
        pygame.draw.rect(screen, color, (i*kw, by, kw-1, kh))
        pygame.draw.rect(screen, BLACK, (i*kw, by, kw-1, kh), 1)
    black_map = {1:0.7, 3:1.7, 6:3.7, 8:4.7, 10:5.7, 13:7.7, 15:8.7, 18:10.7, 20:11.7, 22:12.7}
    for off, pos in black_map.items():
        note = 48 + off + octave_offset
        color = (30, 150, 200) if note in active_notes else BLACK
        pygame.draw.rect(screen, color, (int(pos*kw), by, int(kw//1.6), int(kh//1.6)))

# 5. 메인 루프
clock = pygame.time.Clock()
running, is_dragging_knob = True, -1

while running:
    screen.fill((20, 20, 25))
    now = pygame.time.get_ticks()
    
    if is_playing:
        if now - last_tick > (60000 / tempo) / 4:
            current_step = (current_step + 1) % 16
            last_tick = now
            for r in range(9): 
                if drum_grid[r, current_step]: play_drum_synth(r)
            for r in range(12):
                if bass_grid[r, current_step]: engine_bass(36+(11-r)).play()
                if synth_grid[r, current_step]: engine_synth(48+(11-r)).play()

    if inport:
        for msg in inport.iter_pending():
            note = msg.note + octave_offset
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes.add(note); engine_synth(note).play()
            elif msg.type in ['note_off', 'note_on']: active_notes.discard(note)

    # 상단 컨트롤바
    exit_rect = pygame.Rect(SW - 60, 20, 40, 40)
    pygame.draw.rect(screen, EXIT_RED, exit_rect, border_radius=5)
    screen.blit(large_font.render("X", True, WHITE), (SW-48, 28))
    
    play_rect, stop_rect = pygame.Rect(SW//2-230, 25, 60, 40), pygame.Rect(SW//2-160, 25, 60, 40)
    pygame.draw.rect(screen, PLAY_GREEN if is_playing else (60,80,60), play_rect, border_radius=5)
    pygame.draw.rect(screen, (80,60,60), stop_rect, border_radius=5)
    screen.blit(small_font.render("PLAY", True, WHITE), (play_rect.x+15, 38))
    screen.blit(small_font.render("STOP", True, WHITE), (stop_rect.x+15, 38))
    
    t_m, t_p = pygame.Rect(SW//2-80, 25, 35, 40), pygame.Rect(SW//2+85, 25, 35, 40)
    pygame.draw.rect(screen, GRAY, t_m, border_radius=5); pygame.draw.rect(screen, GRAY, t_p, border_radius=5)
    screen.blit(large_font.render("-", True, WHITE), (t_m.x+11, t_m.y+7))
    screen.blit(large_font.render("+", True, WHITE), (t_p.x+10, t_p.y+8))
    screen.blit(tempo_font.render(f"{int(tempo)}", True, GREEN), (SW//2-30, 28))

    # 그리드 및 악기 섹션
    mw, mh = (SW - 160) // 3, 210
    my = SH // 2 - 35
    random_btns = []

    for i, (name, rows) in enumerate([("DRUM", 9), ("BASS", 12), ("SYNTH", 12)]):
        x = 60 + i*(mw+45)
        
        # 옥타브 버튼 섹션
        if name in ["BASS", "SYNTH"]:
            current_oct = bass_octave if name == "BASS" else synth_octave
            oct_list = [-12, 0, 12] if name == "BASS" else [-24, -12, 0, 12, 24]
            for idx, val in enumerate(oct_list):
                oct_rect = pygame.Rect(x + (idx*35), my-135, 32, 22)
                is_active = (current_oct == val)
                pygame.draw.rect(screen, POINT_COLOR if is_active else GRAY, oct_rect, border_radius=5)
                txt = f"{val:+}" if val != 0 else "0"
                screen.blit(small_font.render(txt, True, WHITE), (oct_rect.x+3, oct_rect.y+5))

        # SYNTH 전용 오실레이터 그림 및 LFO
        if name == "SYNTH":
            for idx, o_type in enumerate(OSC_TYPES):
                btn_rect = pygame.Rect(x + 185 + (idx*55), my-135, 50, 22)
                is_active = synth_settings["SYNTH"]["osc"] == o_type
                pygame.draw.rect(screen, POINT_COLOR if is_active else GRAY, btn_rect, border_radius=5)
                # 파형 아이콘 그리기
                screen.blit(wave_icons[o_type], (btn_rect.x+5, btn_rect.y-16))
                screen.blit(small_font.render(o_type.upper(), True, WHITE), (btn_rect.x+8, btn_rect.y+5))
            
            for j, (label, key) in enumerate([("RAT", "lfo_freq"), ("DEP", "lfo_depth")]):
                lx, ly = x + mw - 70 + (j*40), my-130
                val = synth_settings["SYNTH"][key]
                norm_val = val / 10.0 if key == "lfo_freq" else val / 0.1
                pygame.draw.rect(screen, GRAY, (lx, ly, 6, 40))
                pygame.draw.rect(screen, GREEN, (lx-7, ly+40-(norm_val*40)-3, 20, 6))
                screen.blit(small_font.render(label, True, WHITE), (lx-5, ly+43))

        # ADSR 슬라이더
        if name != "DRUM":
            for j, k in enumerate(["A", "D", "S", "R"]):
                sx, sy = x + (j*35), my-85
                val = adsr_settings[name][k]
                pygame.draw.rect(screen, GRAY, (sx, sy, 6, 45))
                pygame.draw.rect(screen, POINT_COLOR, (sx-7, sy+45-(val*45)-3, 20, 6))
                screen.blit(small_font.render(k, True, WHITE), (sx-1, sy+48))
        
        screen.blit(large_font.render(name, True, WHITE), (x, my-25))
        rbtn = pygame.Rect(x + mw - 70, my-28, 70, 20)
        random_btns.append(rbtn); pygame.draw.rect(screen, (70,70,80), rbtn, border_radius=4)
        screen.blit(small_font.render("RANDOM", True, WHITE), (rbtn.x+10, rbtn.y+3))

        g_data = drum_grid if name=="DRUM" else (bass_grid if name=="BASS" else synth_grid)
        for r in range(rows):
            if name=="DRUM":
                kx, ky = x-25, my + r*(mh/9) + 12
                pygame.draw.circle(screen, GRAY, (int(kx), int(ky)), 12)
                rad = np.radians((drum_pitches[r]/12.0)*135 - 90)
                pygame.draw.line(screen, GREEN, (kx, ky), (kx+np.cos(rad)*12, ky+np.sin(rad)*12), 3)
                screen.blit(small_font.render(DRUM_NAMES[r][:3], True, WHITE), (x-50, my+r*(mh/9)))
            for c in range(16):
                rect = pygame.Rect(x + c*(mw/16), my + r*(mh/rows), (mw/16)-1, (mh/rows)-1)
                color = POINT_COLOR if g_data[r,c] else (40,40,45)
                pygame.draw.rect(screen, (200,200,220) if is_playing and c == current_step else color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

    draw_keyboard()

    # 미디 포트 드롭다운
    drop_rect = pygame.Rect(30, 25, 180, 35)
    pygame.draw.rect(screen, (50,50,60), drop_rect, border_radius=5)
    screen.blit(font.render(selected_port_name[:15], True, WHITE), (40, 33))
    if is_dropdown_open:
        for idx, p in enumerate(input_ports):
            r = pygame.Rect(30, 60 + idx*30, 180, 30)
            pygame.draw.rect(screen, (70,70,80), r)
            screen.blit(font.render(p[:15], True, WHITE), (40, 65 + idx*30))

    # 6. 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if is_dropdown_open:
                for idx, p in enumerate(input_ports):
                    if pygame.Rect(30, 60+idx*30, 180, 30).collidepoint(event.pos):
                        selected_port_name = p; is_dropdown_open = False
                        try:
                            if inport: inport.close()
                            inport = mido.open_input(p)
                        except: pass
                if not drop_rect.collidepoint(event.pos): is_dropdown_open = False
                continue

            if drop_rect.collidepoint(event.pos): is_dropdown_open = True; continue
            if exit_rect.collidepoint(event.pos): running = False
            if play_rect.collidepoint(event.pos): is_playing = True
            if stop_rect.collidepoint(event.pos): is_playing = False; current_step = 0
            if t_p.collidepoint(event.pos): tempo += 1
            if t_m.collidepoint(event.pos): tempo = max(20, tempo - 1)
            
            for r in range(9):
                if pygame.Rect(60-40, my+r*(mh/9)-5, 40, 40).collidepoint(event.pos): is_dragging_knob = r

            for idx, (name, rows) in enumerate([("DRUM", 9), ("BASS", 12), ("SYNTH", 12)]):
                bx = 60 + idx*(mw+45)
                
                # 옥타브 버튼 클릭
                if name in ["BASS", "SYNTH"]:
                    oct_list = [-12, 0, 12] if name == "BASS" else [-24, -12, 0, 12, 24]
                    for o_idx, val in enumerate(oct_list):
                        if pygame.Rect(bx + (o_idx*35), my-135, 32, 22).collidepoint(event.pos):
                            if name == "BASS": bass_octave = val
                            else: synth_octave = val

                if name == "SYNTH":
                    for o_idx, o_type in enumerate(OSC_TYPES):
                        if pygame.Rect(bx + 185 + (o_idx*55), my-135, 50, 22).collidepoint(event.pos):
                            synth_settings["SYNTH"]["osc"] = o_type
                    for j, (label, key) in enumerate([("RAT", "lfo_freq"), ("DEP", "lfo_depth")]):
                        if pygame.Rect(bx + mw - 70 + (j*40)-10, my-130, 26, 40).collidepoint(event.pos):
                            new_val = max(0.0, min(1.0, (my-90 - event.pos[1]) / 40))
                            if key == "lfo_freq": synth_settings["SYNTH"][key] = new_val * 10.0
                            else: synth_settings["SYNTH"][key] = new_val * 0.1

                if name != "DRUM":
                    for j, k in enumerate(["A", "D", "S", "R"]):
                        if pygame.Rect(bx+(j*35)-10, my-85, 26, 45).collidepoint(event.pos):
                            adsr_settings[name][k] = max(0.01, min(1.0, (my-40 - event.pos[1]) / 45))
                
                if my <= event.pos[1] <= my + mh and bx <= event.pos[0] <= bx + mw:
                    r_idx, c_idx = int((event.pos[1]-my)//(mh/rows)), int((event.pos[0]-bx)//(mw/16))
                    if name=="DRUM": drum_grid[r_idx, c_idx]^=1; play_drum_synth(r_idx)
                    elif name=="BASS": bass_grid[r_idx, c_idx]^=1; engine_bass(36+(11-r_idx)).play()
                    else: synth_grid[r_idx, c_idx]^=1; engine_synth(48+(11-r_idx)).play()
            
            for i, rb in enumerate(random_btns):
                if rb.collidepoint(event.pos):
                    target = [drum_grid, bass_grid, synth_grid][i]
                    target.fill(0); [target.__setitem__((random.randint(0, target.shape[0]-1), c), 1) for c in range(16) if random.random()>0.7]

            if event.pos[1] >= SH - 150:
                kw = SW // 14
                white_idx = event.pos[0] // kw
                if white_idx < 14:
                    notes = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21, 23]
                    note = 48 + notes[white_idx] + octave_offset
                    engine_synth(note).play(); active_notes.add(note)

        if event.type == pygame.MOUSEBUTTONUP: 
            is_dragging_knob = -1
            active_notes.clear()
        if event.type == pygame.MOUSEMOTION and is_dragging_knob != -1:
            drum_pitches[is_dragging_knob] = max(-12, min(12, drum_pitches[is_dragging_knob] - event.rel[1]))

    pygame.display.flip()
    clock.tick(60)
pygame.quit()